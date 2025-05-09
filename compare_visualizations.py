#!/usr/bin/env python3
"""
Comparative visualization script for poker evaluation results.

This script generates side-by-side visualizations comparing baseline and SFT model results.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import pandas as pd

def load_metrics(metrics_file):
    """Load metrics from a JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)

def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def compare_overall_accuracy(baseline_metrics, sft_metrics, output_dir):
    """Create a comparative bar chart for overall accuracy metrics."""
    plt.figure(figsize=(12, 7))
    
    # Extract the main metrics for both models
    metrics = {
        'Average Accuracy': [baseline_metrics['average_accuracy'], sft_metrics['average_accuracy']],
        'Action Accuracy': [baseline_metrics['action_accuracy'], sft_metrics['action_accuracy']]
    }
    
    if 'value_accuracy' in baseline_metrics and 'value_accuracy' in sft_metrics:
        metrics['Value Accuracy\n(for bet/raise)'] = [baseline_metrics['value_accuracy'], sft_metrics['value_accuracy']]
    
    # Prepare data for grouped bar chart
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    baseline_bars = ax.bar(x - width/2, [m[0] for m in metrics.values()], width, label='Baseline', color='skyblue')
    sft_bars = ax.bar(x + width/2, [m[1] for m in metrics.values()], width, label='After SFT', color='coral')
    
    # Add percentage labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=11)
    
    add_labels(baseline_bars)
    add_labels(sft_bars)
    
    # Customize the chart
    ax.set_ylabel('Accuracy')
    ax.set_title('Overall Accuracy: Baseline vs SFT Model')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys())
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max([m[1] for m in metrics.values()]) * 1.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Calculate and display improvement
    for i, (metric_name, values) in enumerate(metrics.items()):
        improvement = values[1] - values[0]
        ax.text(i, values[1] + 0.04, f'+{improvement:.1%}', ha='center', va='bottom', 
                color='green', fontweight='bold', fontsize=11)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_overall_accuracy.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'comparison_overall_accuracy.pdf'))
    plt.close()

def compare_action_accuracy(baseline_metrics, sft_metrics, output_dir):
    """Create a comparative bar chart for action-specific accuracy."""
    # Extract action-specific metrics
    baseline_action_metrics = baseline_metrics['action_metrics']
    sft_action_metrics = sft_metrics['action_metrics']
    
    # Define the actions and their order 
    actions = ['check', 'call', 'fold', 'bet', 'raise']
    actions = [a for a in actions if a in baseline_action_metrics and a in sft_action_metrics]
    
    # Extract accuracies
    baseline_accuracies = [baseline_action_metrics[action]['accuracy'] for action in actions]
    sft_accuracies = [sft_action_metrics[action]['accuracy'] for action in actions]
    
    # Prepare data for grouped bar chart
    x = np.arange(len(actions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    baseline_bars = ax.bar(x - width/2, baseline_accuracies, width, label='Baseline', color='skyblue')
    sft_bars = ax.bar(x + width/2, sft_accuracies, width, label='After SFT', color='coral')
    
    # Add percentage labels on top of bars
    def add_labels(bars, values):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=10)
    
    add_labels(baseline_bars, baseline_accuracies)
    add_labels(sft_bars, sft_accuracies)
    
    # Calculate and display improvement
    for i, action in enumerate(actions):
        improvement = sft_accuracies[i] - baseline_accuracies[i]
        ax.text(i, sft_accuracies[i] + 0.04, f'+{improvement:.1%}', 
                ha='center', va='bottom', color='green', fontweight='bold', fontsize=10)
    
    # Customize the chart
    ax.set_ylabel('Accuracy')
    ax.set_title('Action Accuracy by Type: Baseline vs SFT Model')
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in actions])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(sft_accuracies) * 1.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add counts as text below each group
    for i, action in enumerate(actions):
        count = baseline_action_metrics[action]['count']
        ax.text(i, -0.03, f'n={count}', ha='center', va='top', color='darkblue', fontsize=9)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_action_accuracy.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'comparison_action_accuracy.pdf'))
    plt.close()

def compare_bet_raise_value_accuracy(baseline_metrics, sft_metrics, output_dir):
    """Create comparative bar chart for value accuracy on bet/raise actions."""
    # Extract action metrics
    baseline_action_metrics = baseline_metrics['action_metrics']
    sft_action_metrics = sft_metrics['action_metrics']
    
    # Check if we have bet or raise data
    value_actions = []
    baseline_action_acc = []
    baseline_value_acc = []
    sft_action_acc = []
    sft_value_acc = []
    
    for action in ['bet', 'raise']:
        if (action in baseline_action_metrics and 'value_accuracy' in baseline_action_metrics[action] and
            action in sft_action_metrics and 'value_accuracy' in sft_action_metrics[action]):
            value_actions.append(action)
            baseline_action_acc.append(baseline_action_metrics[action]['accuracy'])
            baseline_value_acc.append(baseline_action_metrics[action]['value_accuracy'])
            sft_action_acc.append(sft_action_metrics[action]['accuracy'])
            sft_value_acc.append(sft_action_metrics[action]['value_accuracy'])
    
    if not value_actions:
        return
    
    # Set up the figure with subplots - one for each action
    fig, axes = plt.subplots(1, len(value_actions), figsize=(14, 7), sharey=True)
    if len(value_actions) == 1:
        axes = [axes]  # Make iterable for consistent handling
    
    width = 0.3
    
    # Create a grouped bar chart for each action
    for i, (action, ax) in enumerate(zip(value_actions, axes)):
        x = np.arange(2)  # Two groups: "Action Accuracy" and "Value Accuracy"
        
        # Plot bars
        baseline_bars = ax.bar(x - width/2, [baseline_action_acc[i], baseline_value_acc[i]], width, label='Baseline', color='skyblue')
        sft_bars = ax.bar(x + width/2, [sft_action_acc[i], sft_value_acc[i]], width, label='After SFT', color='coral')
        
        # Add percentage labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        add_labels(baseline_bars)
        add_labels(sft_bars)
        
        # Add improvement text
        for j in range(2):
            baseline_val = baseline_bars[j].get_height()
            sft_val = sft_bars[j].get_height()
            improvement = sft_val - baseline_val
            ax.text(j, sft_val + 0.03, f'+{improvement:.1%}', 
                    ha='center', va='bottom', color='green', fontweight='bold', fontsize=9)
        
        # Customize the subplot
        ax.set_title(f'{action.capitalize()} Action', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(['Action\nAccuracy', 'Value\nAccuracy'])
        if i == 0:
            ax.set_ylabel('Accuracy')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, max(sft_action_acc) * 1.3)  # Give space for labels
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Add count text
        count = baseline_action_metrics[action]['count']
        ax.text(0.5, -0.05, f'n={count}', ha='center', va='top', 
                transform=ax.transAxes, color='darkblue', fontsize=10)
    
    # Add a common legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               fancybox=True, shadow=True, ncol=2)
    
    # Add an overall title
    fig.suptitle('Bet/Raise: Action vs Value Accuracy Comparison', fontsize=14, y=0.95)
    
    # Save figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'comparison_bet_raise_value.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'comparison_bet_raise_value.pdf'))
    plt.close()

def compare_multiple_actions(baseline_metrics, sft_metrics, output_dir):
    """Create a comparative visualization for multiple actions."""
    # Extract data
    baseline_multiple = baseline_metrics.get('multiple_actions_percentage', 0)
    baseline_single = 1 - baseline_multiple
    
    sft_multiple = sft_metrics.get('multiple_actions_percentage', 0)
    sft_single = 1 - sft_multiple
    
    # Create a grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(2)  # Two groups: Single and Multiple
    width = 0.35
    
    baseline_bars = ax.bar(x - width/2, [baseline_single, baseline_multiple], width, label='Baseline', color='skyblue')
    sft_bars = ax.bar(x + width/2, [sft_single, sft_multiple], width, label='After SFT', color='coral')
    
    # Add percentage labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=11)
    
    add_labels(baseline_bars)
    add_labels(sft_bars)
    
    # Add improvement indicators
    improvements = [sft_single - baseline_single, baseline_multiple - sft_multiple]
    for i, improvement in enumerate(improvements):
        # Use a different color and symbol for each type of change
        color = 'green' if improvement > 0 else 'red'
        sign = '+' if improvement > 0 else ''
        # For multiple actions, we want to show reduction as positive
        if i == 1:
            color = 'green' if improvement < 0 else 'red'
            sign = '-' if improvement < 0 else '+'
            improvement = abs(improvement)
        
        y_pos = max(baseline_bars[i].get_height(), sft_bars[i].get_height()) + 0.05
        ax.text(i, y_pos, f'{sign}{improvement:.1%}', 
                ha='center', va='bottom', color=color, fontweight='bold', fontsize=12)
    
    # Customize the chart
    ax.set_ylabel('Percentage of Predictions')
    ax.set_title('Single vs Multiple Action Predictions')
    ax.set_xticks(x)
    ax.set_xticklabels(['Single Action\nPredictions', 'Multiple Actions\nPredictions'])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_multiple_actions.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'comparison_multiple_actions.pdf'))
    plt.close()

def compare_prediction_distribution(baseline_metrics, sft_metrics, output_dir):
    """Create a comparative visualization of prediction distributions."""
    # Extract distribution data
    baseline_dist = baseline_metrics.get('prediction_distribution', {})
    sft_dist = sft_metrics.get('prediction_distribution', {})
    
    # Get all unique prediction types
    all_types = sorted(set(list(baseline_dist.keys()) + list(sft_dist.keys())))
    
    # Organize data in order: first valid actions, then null/unknown
    standard_order = ['check', 'call', 'fold', 'bet', 'raise', 'null', 'unknown']
    ordered_types = [t for t in standard_order if t in all_types]
    
    # Extract counts, filling in zeros for missing types
    baseline_counts = [baseline_dist.get(t, 0) for t in ordered_types]
    sft_counts = [sft_dist.get(t, 0) for t in ordered_types]
    
    # Convert to percentages
    baseline_total = baseline_metrics['total_examples']
    sft_total = sft_metrics['total_examples']
    
    baseline_pcts = [count / baseline_total * 100 for count in baseline_counts]
    sft_pcts = [count / sft_total * 100 for count in sft_counts]
    
    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(ordered_types))
    width = 0.35
    
    baseline_bars = ax.bar(x - width/2, baseline_pcts, width, label='Baseline', color='skyblue')
    sft_bars = ax.bar(x + width/2, sft_pcts, width, label='After SFT', color='coral')
    
    # Add percentage labels
    def add_labels(bars, counts):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = counts[i]
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{count}\n({height:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    add_labels(baseline_bars, baseline_counts)
    add_labels(sft_bars, sft_counts)
    
    # Customize the chart
    ax.set_ylabel('Percentage of Predictions')
    ax.set_title('Prediction Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in ordered_types])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(max(baseline_pcts), max(sft_pcts)) * 1.2)
    
    # Add a reference line for equal distribution
    num_actions = 5  # check, call, fold, bet, raise
    if all(action in ordered_types for action in ['check', 'call', 'fold', 'bet', 'raise']):
        ideal_pct = 100 / num_actions
        ax.axhline(y=ideal_pct, color='green', linestyle='--', alpha=0.7)
        ax.text(len(ordered_types) - 1, ideal_pct + 0.5, f'Ideal: {ideal_pct:.1f}%', 
                ha='right', va='bottom', color='green', fontsize=9)
    
    # Color the bars for null/unknown differently
    for i, type_name in enumerate(ordered_types):
        if type_name in ['null', 'unknown']:
            baseline_bars[i].set_color('lightgray')
            sft_bars[i].set_color('lightpink')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_prediction_distribution.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'comparison_prediction_distribution.pdf'))
    plt.close()

def create_summary_table(baseline_metrics, sft_metrics, output_dir):
    """Create a summary table image comparing key metrics."""
    # Key metrics to display
    metrics = {
        'Average Accuracy': ('average_accuracy', ':.1%'),
        'Action Accuracy': ('action_accuracy', ':.1%'),
        'Value Accuracy': ('value_accuracy', ':.1%'),
        'Multiple Actions %': ('multiple_actions_percentage', ':.1%'),
    }
    
    # Prepare data
    data = []
    for metric_name, (metric_key, format_str) in metrics.items():
        baseline_val = baseline_metrics.get(metric_key, 0)
        sft_val = sft_metrics.get(metric_key, 0)
        change = sft_val - baseline_val
        
        # Format values as percentages
        baseline_str = f"{baseline_val:.1%}"
        sft_str = f"{sft_val:.1%}"
        
        # For multiple actions, negative change is improvement
        if metric_key == 'multiple_actions_percentage':
            change_str = f"{change:.1%}"
            change_color = 'red' if change > 0 else 'green'
        else:
            change_str = f"+{change:.1%}" if change >= 0 else f"{change:.1%}"
            change_color = 'green' if change > 0 else 'red'
        
        data.append([
            metric_name,
            baseline_str,
            sft_str,
            change_str,
            change_color
        ])
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    colLabels = ['Metric', 'Baseline', 'After SFT', 'Improvement', '']
    table = ax.table(cellText=data, colLabels=colLabels, loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Color the change cells
    for i, row in enumerate(data):
        change_color = row[4]
        table[(i+1, 3)].set_text_props(color=change_color, weight='bold')
    
    # Set column widths
    for j in range(5):
        if j == 0:  # Metric name column
            table.auto_set_column_width(j)
        else:
            table.auto_set_column_width(j)
    
    # Set title
    ax.set_title('Baseline vs SFT Model: Key Metrics Comparison', fontsize=14, pad=20)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'comparison_summary_table.pdf'), bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate all comparative visualizations."""
    parser = argparse.ArgumentParser(description='Generate comparative visualizations for poker evaluation results.')
    parser.add_argument('--baseline', type=str, required=True,
                      help='Path to the baseline metrics JSON file')
    parser.add_argument('--sft', type=str, required=True,
                      help='Path to the SFT metrics JSON file')
    parser.add_argument('--output', type=str, default='comparison_visualizations',
                      help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Load the metrics data
    baseline_metrics = load_metrics(args.baseline)
    sft_metrics = load_metrics(args.sft)
    
    # Create output directory
    output_dir = create_output_dir(args.output)
    
    # Generate visualizations
    compare_overall_accuracy(baseline_metrics, sft_metrics, output_dir)
    compare_action_accuracy(baseline_metrics, sft_metrics, output_dir)
    compare_bet_raise_value_accuracy(baseline_metrics, sft_metrics, output_dir)
    compare_multiple_actions(baseline_metrics, sft_metrics, output_dir)
    compare_prediction_distribution(baseline_metrics, sft_metrics, output_dir)
    create_summary_table(baseline_metrics, sft_metrics, output_dir)
    
    print(f"Comparative visualizations created in {output_dir}")

if __name__ == "__main__":
    main() 