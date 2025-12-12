#!/usr/bin/env python3
"""Simple analysis of comparison study results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the summary data
df = pd.read_csv('../../results/comparison_study_20251007_122439/summary.csv',
                 names=['Config', 'Strategy', 'Final_Performance', 'Duration', 'Timestamp'])

# Remove the empty last row
df = df[df['Config'].notna()]

print("🔬 SCHEDULING STRATEGY COMPARISON RESULTS")
print("=" * 50)

# Convert to numeric
df['Final_Performance'] = pd.to_numeric(df['Final_Performance'], errors='coerce')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Rankings by final performance
print("\n📊 FINAL PERFORMANCE RANKING:")
print("-" * 30)
performance_ranking = df.sort_values('Final_Performance', ascending=False)
for i, row in performance_ranking.iterrows():
    print(f"{i+1}. {row['Strategy']}: {row['Final_Performance']:.1f}")

print(f"\n⏱️  TRAINING EFFICIENCY:")
print("-" * 20)
efficiency_ranking = df.sort_values('Duration', ascending=True)
for i, row in efficiency_ranking.iterrows():
    print(f"{i+1}. {row['Strategy']}: {row['Duration']}s")

# Key insights
best_performance = performance_ranking.iloc[0]
fastest_training = efficiency_ranking.iloc[0]

print(f"\n🎯 KEY INSIGHTS:")
print("-" * 15)
print(f"• Best Performance: {best_performance['Strategy']} ({best_performance['Final_Performance']:.1f})")
print(f"• Fastest Training: {fastest_training['Strategy']} ({fastest_training['Duration']}s)")
print(f"• Reward-Based Performance: {df[df['Strategy'].str.contains('Reward-Based')]['Final_Performance'].iloc[0]:.1f}")

# Performance vs Training Time
reward_based_perf = df[df['Strategy'].str.contains('Reward-Based')]['Final_Performance'].iloc[0]
student_only_perf = df[df['Strategy'].str.contains('Student-Only')]['Final_Performance'].iloc[0]

print(f"\n📈 REWARD-BASED VS BASELINES:")
print("-" * 30)
print(f"• vs Student-Only: {reward_based_perf:.1f} vs {student_only_perf:.1f} ({((reward_based_perf/student_only_perf-1)*100):+.1f}%)")

teacher_only_perf = df[df['Strategy'].str.contains('Teacher-Only')]['Final_Performance'].iloc[0]
print(f"• vs Teacher Upper Bound: {reward_based_perf:.1f} vs {teacher_only_perf:.1f} ({((reward_based_perf/teacher_only_perf)*100):.1f}% of optimal)")

print(f"\n🔄 COMPARISON WITH OTHER STRATEGIES:")
print("-" * 35)
for _, row in df.iterrows():
    if 'Reward-Based' not in row['Strategy']:
        comparison = ((reward_based_perf / row['Final_Performance'] - 1) * 100)
        print(f"• vs {row['Strategy']}: {comparison:+.1f}%")

print(f"\n💡 CONCLUSIONS:")
print("-" * 12)
if reward_based_perf > student_only_perf:
    print("✅ Reward-based scheduling outperforms student-only baseline")
else:
    print("⚠️  Reward-based scheduling underperforms student-only baseline")

if reward_based_perf > 200:
    print("✅ Reward-based achieves good CartPole performance (>200)")
else:
    print("⚠️  Reward-based needs hyperparameter tuning")

# Create visualization
plt.figure(figsize=(12, 8))

# Performance comparison
plt.subplot(2, 2, 1)
sns.barplot(data=df, x='Final_Performance', y='Strategy', palette='viridis')
plt.title('Final Performance by Strategy')
plt.xlabel('Final Performance')

# Training time comparison
plt.subplot(2, 2, 2)
sns.barplot(data=df, x='Duration', y='Strategy', palette='plasma')
plt.title('Training Time by Strategy')
plt.xlabel('Training Time (seconds)')

# Performance vs Time scatter
plt.subplot(2, 2, 3)
plt.scatter(df['Duration'], df['Final_Performance'], s=100, alpha=0.7)
for i, row in df.iterrows():
    plt.annotate(row['Strategy'].replace(' ', '\n'),
                (row['Duration'], row['Final_Performance']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
plt.xlabel('Training Time (s)')
plt.ylabel('Final Performance')
plt.title('Efficiency vs Performance Trade-off')

# Strategy ranking
plt.subplot(2, 2, 4)
strategy_short = [s.split(' ')[0] + (' (RB)' if 'Reward' in s else '') for s in df['Strategy']]
plt.barh(range(len(df)), df['Final_Performance'],
         color=['red' if 'Reward' in s else 'blue' for s in df['Strategy']])
plt.yticks(range(len(df)), strategy_short)
plt.xlabel('Final Performance')
plt.title('Strategy Ranking (Red = Reward-Based)')

plt.tight_layout()
plt.savefig('../../results/comparison_study_20251007_122439/detailed_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Detailed plots saved: ../../results/comparison_study_20251007_122439/detailed_analysis.png")