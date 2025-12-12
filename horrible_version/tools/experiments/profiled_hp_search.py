#!/usr/bin/env python3
"""Hyperparameter search with comprehensive profiling and resource monitoring."""

import subprocess
import time
import pandas as pd
import psutil
import json
import threading
import queue
from pathlib import Path
from datetime import datetime
import random


class ResourceMonitor:
    """Monitor CPU, memory, and system resources during experiment runs."""

    def __init__(self, log_interval=5.0):
        self.log_interval = log_interval
        self.monitoring = False
        self.data = []
        self.thread = None

    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.data = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop_monitoring(self):
        """Stop monitoring and return collected data."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)
        return self.data.copy()

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get current timestamp
                timestamp = time.time()

                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_count = psutil.cpu_count()

                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                memory_percent = memory.percent

                # Disk I/O
                disk_io = psutil.disk_io_counters()

                # Network I/O (if available)
                try:
                    net_io = psutil.net_io_counters()
                    net_bytes_sent = net_io.bytes_sent if net_io else 0
                    net_bytes_recv = net_io.bytes_recv if net_io else 0
                except:
                    net_bytes_sent = net_bytes_recv = 0

                # GPU info (try to get basic info)
                gpu_info = self._get_gpu_info()

                data_point = {
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'memory_mb': memory_mb,
                    'memory_percent': memory_percent,
                    'disk_read_mb': disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
                    'disk_write_mb': disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
                    'net_sent_mb': net_bytes_sent / (1024 * 1024),
                    'net_recv_mb': net_bytes_recv / (1024 * 1024),
                    **gpu_info
                }

                self.data.append(data_point)

            except Exception as e:
                print(f"Warning: Resource monitoring error: {e}")

            time.sleep(self.log_interval)

    def _get_gpu_info(self):
        """Try to get GPU information using nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = lines[0].split(', ')
                    return {
                        'gpu_utilization': float(gpu_util),
                        'gpu_memory_used_mb': float(gpu_mem_used),
                        'gpu_memory_total_mb': float(gpu_mem_total),
                        'gpu_temperature': float(gpu_temp)
                    }
        except:
            pass

        return {
            'gpu_utilization': 0.0,
            'gpu_memory_used_mb': 0.0,
            'gpu_memory_total_mb': 0.0,
            'gpu_temperature': 0.0
        }


def profile_experiment(config, experiment_id, results_dir):
    """Run experiment with comprehensive profiling."""

    start_time = time.time()
    exp_name = f"hp_{config['name']}_{experiment_id:02d}"
    log_file = results_dir / f"{exp_name}.log"
    profile_file = results_dir / f"{exp_name}_profile.json"

    # Start resource monitoring
    monitor = ResourceMonitor(log_interval=2.0)  # Sample every 2 seconds

    try:
        # Get initial system state
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)
        initial_cpu = psutil.cpu_percent(interval=1)

        print(f"🚀 Starting {exp_name}: {config['description']}")
        print(f"   Parameters: timesteps={config['total_timesteps']}, lr={config['learning_rate']}")
        print(f"   Initial Memory: {initial_memory:.1f}MB, CPU: {initial_cpu:.1f}%")

        # Start monitoring
        monitor.start_monitoring()

        # Build command
        cmd = [
            "../../run_experiment.sh",
            "--config-path", "configs/experiments/comprehensive",
            "--config-name", "cartpole_reward_based",
            f"total_timesteps={config['total_timesteps']}",
            f"learning_rate={config['learning_rate']}",
            f"seed={random.randint(1, 9999)}"
        ]

        # Run experiment with resource tracking
        experiment_start = time.time()

        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

            # Monitor process
            try:
                process.wait(timeout=2400)  # 40 minute timeout
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = -1
                print(f"⏰ {exp_name} timed out and was killed")

        experiment_end = time.time()
        experiment_duration = experiment_end - experiment_start

        # Stop monitoring and get data
        resource_data = monitor.stop_monitoring()

        # Get final system state
        final_memory = psutil.virtual_memory().used / (1024 * 1024)
        final_cpu = psutil.cpu_percent(interval=1)

        # Parse experiment output
        final_performance = 0.0
        training_time = 0.0
        total_episodes = 0

        if return_code == 0:
            with open(log_file, 'r') as f:
                output = f.read()

            for line in output.split('\n'):
                if "Final performance:" in line:
                    try:
                        final_performance = float(line.split(':')[1].strip())
                    except:
                        pass
                if "Training completed in" in line:
                    try:
                        # Extract training time
                        time_str = line.split('in')[1].split('s')[0].strip()
                        training_time = float(time_str)
                        # Extract episodes if available
                        if "episodes" in line:
                            episodes_str = line.split('(')[1].split('episodes')[0].strip()
                            total_episodes = int(episodes_str.split()[-1])
                    except:
                        pass

        # Analyze resource usage
        resource_stats = {}
        if resource_data:
            df_resources = pd.DataFrame(resource_data)
            resource_stats = {
                'avg_cpu_percent': df_resources['cpu_percent'].mean(),
                'max_cpu_percent': df_resources['cpu_percent'].max(),
                'avg_memory_mb': df_resources['memory_mb'].mean(),
                'max_memory_mb': df_resources['memory_mb'].max(),
                'avg_memory_percent': df_resources['memory_percent'].mean(),
                'max_memory_percent': df_resources['memory_percent'].max(),
                'total_disk_read_mb': df_resources['disk_read_mb'].iloc[-1] - df_resources['disk_read_mb'].iloc[0] if len(df_resources) > 1 else 0,
                'total_disk_write_mb': df_resources['disk_write_mb'].iloc[-1] - df_resources['disk_write_mb'].iloc[0] if len(df_resources) > 1 else 0,
                'avg_gpu_utilization': df_resources['gpu_utilization'].mean(),
                'max_gpu_utilization': df_resources['gpu_utilization'].max(),
                'avg_gpu_memory_mb': df_resources['gpu_memory_used_mb'].mean(),
                'max_gpu_memory_mb': df_resources['gpu_memory_used_mb'].max(),
                'max_gpu_temperature': df_resources['gpu_temperature'].max(),
            }

        # Compile comprehensive results
        total_duration = time.time() - start_time

        result_data = {
            # Experiment info
            'experiment_id': experiment_id,
            'name': config['name'],
            'description': config['description'],
            'timestamp': datetime.now().isoformat(),

            # Performance metrics
            'final_performance': final_performance,
            'total_episodes': total_episodes,
            'success': return_code == 0,

            # Timing metrics
            'experiment_duration_s': experiment_duration,
            'training_time_s': training_time,
            'total_duration_s': total_duration,
            'overhead_s': total_duration - experiment_duration,

            # Resource usage
            'memory_delta_mb': final_memory - initial_memory,
            'cpu_efficiency': final_performance / resource_stats.get('avg_cpu_percent', 1) if resource_stats.get('avg_cpu_percent', 0) > 0 else 0,
            'memory_efficiency': final_performance / resource_stats.get('avg_memory_percent', 1) if resource_stats.get('avg_memory_percent', 0) > 0 else 0,

            # Configuration
            'total_timesteps': config['total_timesteps'],
            'learning_rate': config['learning_rate'],

            # Resource stats
            **resource_stats
        }

        # Save detailed profiling data
        profile_data = {
            'result': result_data,
            'resource_timeline': resource_data,
            'config': config
        }

        with open(profile_file, 'w') as f:
            json.dump(profile_data, f, indent=2)

        # Status output
        status = "✅" if return_code == 0 else "❌"
        print(f"{status} {exp_name}: {final_performance:.1f} points")
        print(f"   Duration: {experiment_duration:.1f}s, Memory: {final_memory-initial_memory:+.1f}MB")
        if resource_stats:
            print(f"   Avg CPU: {resource_stats['avg_cpu_percent']:.1f}%, Peak Memory: {resource_stats['max_memory_percent']:.1f}%")
            if resource_stats['max_gpu_utilization'] > 0:
                print(f"   GPU Utilization: {resource_stats['avg_gpu_utilization']:.1f}% (peak: {resource_stats['max_gpu_utilization']:.1f}%)")

        return result_data

    except Exception as e:
        monitor.stop_monitoring()
        print(f"❌ {exp_name} failed with error: {e}")
        return {
            'experiment_id': experiment_id,
            'name': config['name'],
            'final_performance': 0.0,
            'success': False,
            'error': str(e),
            'total_duration_s': time.time() - start_time,
            **config
        }


def create_search_space():
    """Create search space for hyperparameter optimization."""

    configurations = [
        # Quick baseline tests
        {'name': 'quick_test', 'total_timesteps': 50000, 'learning_rate': 3e-4, 'description': 'Quick baseline test'},

        # Extended training variations
        {'name': 'extended_standard', 'total_timesteps': 200000, 'learning_rate': 3e-4, 'description': 'Extended training, standard LR'},
        {'name': 'extended_low_lr', 'total_timesteps': 200000, 'learning_rate': 1e-4, 'description': 'Extended training, low LR'},
        {'name': 'extended_high_lr', 'total_timesteps': 200000, 'learning_rate': 5e-4, 'description': 'Extended training, high LR'},

        # Very long training
        {'name': 'very_long_standard', 'total_timesteps': 300000, 'learning_rate': 3e-4, 'description': 'Very long training, standard LR'},
        {'name': 'very_long_low_lr', 'total_timesteps': 300000, 'learning_rate': 1e-4, 'description': 'Very long training, low LR'},

        # Learning rate variations
        {'name': 'very_low_lr', 'total_timesteps': 150000, 'learning_rate': 5e-5, 'description': 'Very low learning rate'},
        {'name': 'high_lr', 'total_timesteps': 150000, 'learning_rate': 1e-3, 'description': 'High learning rate'},

        # Conservative approach (longer training, lower LR)
        {'name': 'conservative', 'total_timesteps': 400000, 'learning_rate': 5e-5, 'description': 'Conservative: very long + very low LR'},
    ]

    return configurations


def main():
    print("🔬 Profiled Hyperparameter Search for Reward-Based Scheduling")
    print("=" * 65)
    print(f"🖥️  System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total/(1024**3):.1f}GB RAM")

    # Check GPU availability
    try:
        gpu_result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        if gpu_result.returncode == 0:
            gpu_count = len(gpu_result.stdout.strip().split('\n'))
            print(f"🎮 GPU: {gpu_count} NVIDIA GPU(s) detected")
        else:
            print("🎮 GPU: Not available")
    except:
        print("🎮 GPU: Not available")

    # Create results directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/profiled_hp_search_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Results: {results_dir}")

    # Get search space
    search_space = create_search_space()
    print(f"📊 Testing {len(search_space)} configurations")
    print()

    # Run experiments
    results = []
    overall_start = time.time()

    for i, config in enumerate(search_space):
        print(f"--- Experiment {i+1}/{len(search_space)} ---")

        result = profile_experiment(config, i, results_dir)
        results.append(result)

        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(results_dir / "results.csv", index=False)

        print()

    # Final analysis
    print("🎯 FINAL ANALYSIS")
    print("=" * 40)

    df = pd.DataFrame(results)
    successful = df[df['success'] == True]

    if len(successful) > 0:
        # Performance ranking
        top_configs = successful.nlargest(5, 'final_performance')

        print("🏆 TOP CONFIGURATIONS BY PERFORMANCE:")
        for i, (_, row) in enumerate(top_configs.iterrows(), 1):
            print(f"{i}. {row['name']}: {row['final_performance']:.1f}")
            print(f"   {row['description']}")
            print(f"   Training: {row.get('training_time_s', 0):.1f}s, Total: {row.get('total_duration_s', 0):.1f}s")
            if 'avg_cpu_percent' in row:
                print(f"   CPU: {row['avg_cpu_percent']:.1f}%, Memory: {row['avg_memory_percent']:.1f}%")
            print()

        # Efficiency analysis
        print("⚡ EFFICIENCY ANALYSIS:")
        df_eff = successful.copy()
        df_eff['time_efficiency'] = df_eff['final_performance'] / df_eff['total_duration_s']
        df_eff['resource_efficiency'] = df_eff['final_performance'] / (df_eff.get('avg_cpu_percent', 100) * df_eff.get('avg_memory_percent', 100) / 10000)

        most_time_efficient = df_eff.loc[df_eff['time_efficiency'].idxmax()]
        print(f"🕒 Most time efficient: {most_time_efficient['name']} ({most_time_efficient['time_efficiency']:.3f} points/second)")

        if 'avg_cpu_percent' in df_eff.columns and df_eff['avg_cpu_percent'].sum() > 0:
            most_resource_efficient = df_eff.loc[df_eff['resource_efficiency'].idxmax()]
            print(f"💻 Most resource efficient: {most_resource_efficient['name']} ({most_resource_efficient['resource_efficiency']:.3f})")

        # Baseline comparison
        baseline_perf = 195.8
        best_perf = top_configs.iloc[0]['final_performance']
        print(f"\n📊 VS BASELINE:")
        print(f"   Best result: {best_perf:.1f}")
        print(f"   Student baseline: {baseline_perf:.1f}")
        print(f"   Improvement: {((best_perf/baseline_perf-1)*100):+.1f}%")

        if best_perf > baseline_perf:
            print("🎉 SUCCESS! Found better configuration!")

    else:
        print("❌ No successful experiments")

    total_time = (time.time() - overall_start) / 3600
    print(f"\n⏱️  Total search time: {total_time:.2f} hours")
    print(f"💾 Detailed profiles saved in: {results_dir}")
    print(f"📄 Main results: {results_dir}/results.csv")


if __name__ == "__main__":
    main()