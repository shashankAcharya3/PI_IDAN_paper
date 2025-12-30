"""
Visualization Module for PI-CDA Results
Creates diagrams showing physics-informed learning improvements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150


def load_results():
    """Load comparison results from CSV"""
    comparison_path = "results/comparison_results.csv"
    metrics_path = "results/complete_metrics.csv"
    
    comparison_df = None
    metrics_df = None
    
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        print(f"Loaded comparison results: {len(comparison_df)} rows")
    else:
        print(f"Warning: {comparison_path} not found. Run run_comparison.py first.")
    
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        print(f"Loaded complete metrics: {len(metrics_df)} rows")
    
    return comparison_df, metrics_df


def plot_accuracy_comparison(df, save_path="results/accuracy_comparison.png"):
    """
    Diagram 1: Line plot showing accuracy over batches for PI-CDA vs Baseline
    """
    if df is None:
        print("No comparison data available for accuracy comparison plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate data by method (handle both PI-CDA and PI-IDAN names)
    methods = df['Method'].unique()
    physics_method = [m for m in methods if m != 'Baseline'][0] if len(methods) > 1 else methods[0]
    
    picda = df[df['Method'] == physics_method]
    baseline = df[df['Method'] == 'Baseline']
    
    # Plot lines
    ax.plot(picda['Batch'], picda['Accuracy'], 
            marker='o', linewidth=2.5, markersize=8,
            color='#2E86AB', label='PI-CDA (Physics-Informed)')
    ax.plot(baseline['Batch'], baseline['Accuracy'], 
            marker='s', linewidth=2.5, markersize=8,
            color='#E94F37', label='Baseline (No Physics)')
    
    # Fill area between curves to show improvement (only if both have data)
    if len(picda) > 0 and len(baseline) > 0 and len(picda) == len(baseline):
        ax.fill_between(picda['Batch'], baseline['Accuracy'].values, picda['Accuracy'].values,
                        alpha=0.2, color='#2E86AB', where=picda['Accuracy'].values >= baseline['Accuracy'].values,
                        label='Physics Improvement')
    
    ax.set_xlabel('Batch Number (Temporal Drift)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Sensor Drift Compensation: {physics_method} vs Baseline')
    ax.legend(loc='lower left')
    ax.set_xticks(range(3, 11))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_metrics_per_batch(df, save_path="results/metrics_per_batch.png"):
    """
    Diagram 2: Grouped bar chart showing PR%, RC%, F1% per batch
    """
    if df is None:
        print("No metrics data available for metrics per batch plot")
        return
    
    # Use physics-informed results only for this chart (handle both PI-CDA and PI-IDAN)
    if 'Method' in df.columns:
        methods = df['Method'].unique()
        physics_method = [m for m in methods if m != 'Baseline'][0] if len(methods) > 1 else methods[0]
        df = df[df['Method'] == physics_method].copy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax.bar(x - width, df['Precision'], width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, df['Recall'], width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, df['F1'], width, label='F1-Score', color='#9b59b6')
    
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Score (%)')
    ax.set_title('PI-IDAN Performance Metrics per Batch')
    ax.set_xticks(x)
    ax.set_xticklabels([f'B{b}' for b in df['Batch']])
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_improvement_delta(df, save_path="results/improvement_delta.png"):
    """
    Diagram 3: Bar chart showing improvement (PI-CDA - Baseline) per batch
    """
    if df is None:
        print("No comparison data available for improvement delta plot")
        return
    
    # Calculate improvement (handle both PI-CDA and PI-IDAN)
    methods = df['Method'].unique()
    physics_method = [m for m in methods if m != 'Baseline'][0] if len(methods) > 1 else methods[0]
    
    picda = df[df['Method'] == physics_method].set_index('Batch')
    baseline = df[df['Method'] == 'Baseline'].set_index('Batch')
    
    improvement = picda[['Accuracy', 'Precision', 'Recall', 'F1']] - baseline[['Accuracy', 'Precision', 'Recall', 'F1']]
    improvement = improvement.reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(improvement))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, improvement['Accuracy'], width, label='Accuracy', color='#e74c3c')
    bars2 = ax.bar(x - 0.5*width, improvement['Precision'], width, label='Precision', color='#3498db')
    bars3 = ax.bar(x + 0.5*width, improvement['Recall'], width, label='Recall', color='#2ecc71')
    bars4 = ax.bar(x + 1.5*width, improvement['F1'], width, label='F1-Score', color='#9b59b6')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Batch Number')
    ax.set_ylabel(f'Improvement ({physics_method} - Baseline) (%)')
    ax.set_title('Physics-Informed Learning Improvement Over Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([f'B{b}' for b in improvement['Batch']])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color bars based on positive/negative
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            if bar.get_height() < 0:
                bar.set_alpha(0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_summary_comparison(df, save_path="results/summary_comparison.png"):
    """
    Diagram 4: Summary bar chart showing average metrics for PI-CDA vs Baseline
    """
    if df is None:
        print("No comparison data available for summary comparison plot")
        return
    
    # Calculate averages (handle both PI-CDA and PI-IDAN)
    methods = df['Method'].unique()
    physics_method = [m for m in methods if m != 'Baseline'][0] if len(methods) > 1 else methods[0]
    
    picda = df[df['Method'] == physics_method][['Accuracy', 'Precision', 'Recall', 'F1']].mean()
    baseline = df[df['Method'] == 'Baseline'][['Accuracy', 'Precision', 'Recall', 'F1']].mean()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, [picda[m] for m in metrics], width, 
                   label=f'{physics_method} (Physics-Informed)', color='#2E86AB')
    bars2 = ax.bar(x + width/2, [baseline[m] for m in metrics], width, 
                   label='Baseline (No Physics)', color='#E94F37')
    
    ax.set_ylabel('Score (%)')
    ax.set_title(f'Average Performance: {physics_method} vs Baseline (All Batches)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_all_visualizations():
    """Generate all visualizations"""
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)
    
    os.makedirs("results", exist_ok=True)
    
    comparison_df, metrics_df = load_results()
    
    # Generate all plots
    plot_accuracy_comparison(comparison_df)
    plot_metrics_per_batch(comparison_df if comparison_df is not None else metrics_df)
    plot_improvement_delta(comparison_df)
    plot_summary_comparison(comparison_df)
    
    print("\n" + "=" * 50)
    print("VISUALIZATION COMPLETE")
    print("=" * 50)
    
    # List generated files
    result_files = [f for f in os.listdir("results") if f.endswith('.png')]
    if result_files:
        print("\nGenerated diagrams:")
        for f in sorted(result_files):
            print(f"  - results/{f}")


if __name__ == "__main__":
    create_all_visualizations()
