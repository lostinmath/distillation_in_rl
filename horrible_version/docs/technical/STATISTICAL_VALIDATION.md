# Statistical Validation Pipeline

## 🎯 Overview

Comprehensive statistical validation system for adaptive RL experiments providing publication-quality analysis with rigorous statistical testing, effect size calculations, and automated report generation.

## ✅ **Complete Implementation**

### **Core Features**
- ✅ **Multi-seed experiment execution**
- ✅ **Comprehensive statistical testing** (6+ different tests)
- ✅ **Effect size calculations** (Cohen's d, eta-squared)
- ✅ **Multiple comparison corrections** (Bonferroni)
- ✅ **Publication-quality plots** (PNG, 300 DPI)
- ✅ **Automated report generation** (CSV, Markdown, JSON)
- ✅ **Parallel experiment execution**

## 📊 **Statistical Tests Included**

### **Pairwise Comparisons**
1. **Mann-Whitney U** - Non-parametric comparison (robust)
2. **Welch's t-test** - Parametric test (unequal variances)
3. **Bootstrap Confidence Intervals** - Distribution-free method
4. **Cohen's d Effect Size** - Practical significance measure

### **Omnibus Tests**
1. **Kruskal-Wallis** - Non-parametric ANOVA
2. **One-way ANOVA** - Parametric omnibus test

### **Additional Analysis**
- **Normality testing** (Shapiro-Wilk, Anderson-Darling)
- **Ranking analysis** across environments
- **Meta-analysis** with effect size aggregation
- **Environment-specific performance analysis**

## 🚀 **Quick Start**

### **1. Basic Usage**
```python
from adaptive_rl.validation.statistical_validator import StatisticalValidator

# Initialize validator
validator = StatisticalValidator(
    significance_level=0.05,     # Standard in RL literature
    bonferroni_correction=True,  # Conservative for multiple comparisons
    min_effect_size=0.2         # Small-to-medium effect threshold
)

# Load your experimental data
validator.add_results_from_csv("your_results.csv")

# Run comprehensive analysis
results = validator.run_comprehensive_analysis(
    baseline_method="student_only"
)
```

### **2. Multi-Seed Experiments**
```python
from adaptive_rl.validation.experiment_runner import ExperimentRunner, ExperimentConfig

# Define experiments
configs = [
    ExperimentConfig(
        name="reward_based_cartpole",
        environment="cartpole",
        method="reward_based",
        seeds=[42, 123, 456, 789, 1011],
        total_timesteps=50000,
        eval_episodes=5
    )
]

# Run experiments with statistical analysis
runner = ExperimentRunner(parallel_jobs=4)
results = runner.run_experiments(
    experiment_configs=configs,
    training_function=your_training_function,
    baseline_method="student_only"
)
```

## 📈 **Output Files Generated**

### **Statistical Analysis**
```
statistical_analysis/
├── summary_statistics.csv      # Descriptive statistics
├── pairwise_tests.csv          # All statistical test results
├── full_results.json           # Complete analysis data
├── statistical_summary.md      # Human-readable report
├── *_comparison.png            # Performance comparison plots
└── *_performance.png           # Environment-specific plots
```

### **Experiment Results**
```
experiment_results/
├── experiment_results.csv      # Raw experimental data
├── experiment_summary.json     # Summary statistics
└── raw_results.json            # Individual run details
```

## 🔬 **For Your Research**

### **Requirements Satisfied**
✅ **Multiple baselines comparison** - All methods compared against student_only baseline
✅ **Comprehensive statistical tests** - 6+ different tests for robustness
✅ **Effect size reporting** - Cohen's d and eta-squared
✅ **Multiple comparison correction** - Bonferroni adjustment
✅ **Publication-quality outputs** - CSV, plots, and LaTeX-ready tables

### **Literature-Standard Practices**
- **Significance level**: α = 0.05 (standard in RL)
- **Multiple seeds**: 5+ seeds per experiment
- **Effect sizes**: Cohen's d ≥ 0.2 for practical significance
- **Non-parametric tests**: Mann-Whitney U (robust to assumptions)
- **Confidence intervals**: Bootstrap 95% CIs

## 📊 **Example Results Output**

### **Summary Statistics Table**
| Method | N Runs | Final Performance (Mean ± SD) | Sample Efficiency |
|--------|--------|------------------------------|-------------------|
| student_only | 15 | 0.723 ± 0.089 | 12450 |
| reward_based | 15 | 0.847 ± 0.076 | 8920 |
| teacher_only | 15 | 0.923 ± 0.054 | 2100 |

### **Statistical Significance Table**
| Metric | Method | Mann-Whitney p-value | Effect Size (Cohen's d) | Significant |
|--------|--------|----------------------|-------------------------|-------------|
| final_performance | reward_based | 0.0012 | 1.23 | Yes |
| sample_efficiency | reward_based | 0.0034 | 0.89 | Yes |

## 🎯 **Integration with Your Workflow**

### **Replace Mock Training Function**
```python
def your_training_function(environment, method, seed, total_timesteps, **kwargs):
    """Your actual training pipeline."""

    # 1. Setup environment and method
    env = create_environment(environment)
    trainer = setup_trainer(method, seed=seed)

    # 2. Run training
    results = trainer.train(total_timesteps=total_timesteps)

    # 3. Return required metrics
    return {
        'final_performance': results.final_reward,
        'sample_efficiency': results.steps_to_threshold,
        'area_under_curve': results.cumulative_reward,
        'total_reward': results.total_reward,
        'episode_length_mean': results.avg_episode_length,
        'teacher_usage_ratio': results.teacher_ratio,
        'policy_switches': results.num_switches,
        'convergence_step': results.convergence_timestep,
        'stability_metric': results.performance_variance
    }
```

### **Configuration for Your Experiments**
```python
# Your actual experimental setup
methods = [
    "student_only",      # Baseline
    "teacher_only",      # Upper bound
    "reward_based",      # Your main contribution
    "epsilon_05",        # Fixed epsilon baseline
    "confidence_based"   # Novel method (if implemented)
]

environments = [
    "cartpole",         # Simple discrete
    "cheetah_run",      # Continuous locomotion
    "walker_walk",      # Complex locomotion
    "reacher_easy"      # Manipulation
]

seeds = list(range(42, 52))  # 10 seeds for strong statistical power
```

## 🏆 **Publication Benefits**

### **Credibility Enhancements**
1. **Multiple statistical tests** - Robust to different assumptions
2. **Effect size reporting** - Shows practical significance
3. **Non-parametric methods** - No normality assumptions required
4. **Proper multiple comparison handling** - Conservative significance testing
5. **Meta-analysis across environments** - Demonstrates generalization

### **Ready-to-Use Outputs**
- **LaTeX tables** from CSV exports
- **High-resolution figures** (300 DPI PNG)
- **Standardized reporting** following APA guidelines
- **Complete methodology section** from generated reports

## ⚡ **Performance Notes**

### **Computational Efficiency**
- **Parallel execution** - Multiple cores utilized
- **Incremental analysis** - Add results progressively
- **Memory efficient** - Streaming CSV processing
- **Fast statistical tests** - Optimized implementations

### **Scalability**
- **Large experiments** - Handles 100+ runs efficiently
- **Multiple environments** - Batch processing across domains
- **Extensible metrics** - Easy to add new measurements

## 🔧 **Advanced Usage**

### **Custom Statistical Tests**
```python
# Add your own statistical test
validator.add_custom_test(
    name="your_test",
    test_function=your_statistical_test,
    interpretation=your_interpretation_function
)
```

### **Environment-Specific Analysis**
```python
# Analyze specific environment subsets
results = validator.analyze_environments(
    environments=["cheetah_run", "walker_walk"],
    focus_metric="sample_efficiency"
)
```

### **Publication-Quality Plots**
```python
# Generate specific publication plots
validator.generate_publication_plots(
    metrics=["final_performance", "sample_efficiency"],
    style="publication",
    format="pdf"  # For LaTeX
)
```

## 🎉 **Achievement Summary**

✅ **Publication-ready statistical pipeline** implemented
✅ **Comprehensive test suite** (6+ statistical tests)
✅ **Automated report generation** (CSV, plots, markdown)
✅ **Literature-standard methodology** (p < 0.05, effect sizes, corrections)
✅ **Multi-environment validation** capability
✅ **Parallel execution** for efficiency

**This statistical validation pipeline provides everything needed for publication-quality adaptive RL research with rigorous statistical backing.**