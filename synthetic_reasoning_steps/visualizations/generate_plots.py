#!/usr/bin/env python3
"""
Visualization script for poker evaluation results.

This script generates visualizations for baseline evaluation results
to be included in a research paper.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

def load_metrics(metrics_file):
    """Load metrics from a JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)

def create_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_overall_accuracy(metrics, output_dir):
    """Create a bar chart for overall accuracy metrics."""
    plt.figure(figsize=(10, 6))
    
    # Extract the main metrics
    accuracy_metrics = {
        'Average Accuracy': metrics['average_accuracy'],
        'Action Accuracy': metrics['action_accuracy']
    }
    
    if 'value_accuracy' in metrics:
        accuracy_metrics['Value Accuracy\n(for bet/raise)'] = metrics['value_accuracy']
    
    # Plot bar chart
    bars = plt.bar(accuracy_metrics.keys(), accuracy_metrics.values(), color='skyblue')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', fontsize=12)
    
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy Metrics')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_accuracy.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'overall_accuracy.pdf'))
    plt.close()

def plot_action_accuracy(metrics, output_dir):
    """Create a bar chart for action-specific accuracy."""
    plt.figure(figsize=(12, 7))
    
    # Extract action-specific metrics
    action_metrics = metrics['action_metrics']
    
    # Define the actions and their order 
    actions = ['check', 'call', 'fold', 'bet', 'raise']
    actions = [a for a in actions if a in action_metrics]
    
    # Extract accuracies and counts
    accuracies = [action_metrics[action]['accuracy'] for action in actions]
    counts = [action_metrics[action]['count'] for action in actions]
    
    # Create the primary bar chart for accuracies
    ax1 = plt.gca()
    bars = ax1.bar(actions, accuracies, color='skyblue')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', fontsize=12)
    
    # Configure the primary y-axis for accuracies
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Accuracy')
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Create a secondary y-axis for counts
    ax2 = ax1.twinx()
    ax2.plot(actions, counts, 'ro-', linewidth=2, markersize=8)
    
    # Add count labels
    for i, count in enumerate(counts):
        ax2.text(i, count + 50, f'{count}', ha='center', va='bottom', 
                color='darkred', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Number of Examples', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    # Set title and configure grid
    plt.title('Accuracy by Action Type', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a legend
    from matplotlib.lines import Line2D
    custom_lines = [
        plt.Rectangle((0,0), 1, 1, color='skyblue'),
        Line2D([0], [0], color='darkred', marker='o', markersize=8, linewidth=2)
    ]
    ax1.legend(custom_lines, ['Accuracy', 'Example Count'], loc='upper right')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_accuracy.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'action_accuracy.pdf'))
    plt.close()

def plot_bet_raise_value_accuracy(metrics, output_dir):
    """Create bar chart for value accuracy on bet/raise actions."""
    if 'value_accuracy' not in metrics:
        return
    
    action_metrics = metrics['action_metrics']
    
    # Check if we have bet or raise data
    if 'bet' not in action_metrics and 'raise' not in action_metrics:
        return
        
    plt.figure(figsize=(10, 6))
    
    # Extract data
    actions = []
    action_accuracies = []
    value_accuracies = []
    
    if 'bet' in action_metrics and 'value_accuracy' in action_metrics['bet']:
        actions.append('bet')
        action_accuracies.append(action_metrics['bet']['accuracy'])
        value_accuracies.append(action_metrics['bet']['value_accuracy'])
        
    if 'raise' in action_metrics and 'value_accuracy' in action_metrics['raise']:
        actions.append('raise')
        action_accuracies.append(action_metrics['raise']['accuracy'])
        value_accuracies.append(action_metrics['raise']['value_accuracy'])
    
    # Create grouped bar chart
    x = np.arange(len(actions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, action_accuracies, width, label='Action Accuracy', color='skyblue')
    bars2 = ax.bar(x + width/2, value_accuracies, width, label='Value Accuracy', color='lightcoral')
    
    # Add percentage labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=12)
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Customize the chart
    ax.set_ylabel('Accuracy')
    ax.set_title('Bet/Raise: Action vs Value Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bet_raise_value_accuracy.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'bet_raise_value_accuracy.pdf'))
    plt.close()

def plot_multiple_actions(metrics, output_dir):
    """Create a pie chart showing proportion of predictions with multiple actions."""
    plt.figure(figsize=(10, 6))
    
    # Extract data
    multiple_actions = metrics.get('multiple_actions_count', 0)
    single_action = metrics['total_examples'] - multiple_actions
    
    # Create data for pie chart
    labels = ['Single Action', 'Multiple Actions']
    sizes = [single_action, multiple_actions]
    colors = ['lightgreen', 'lightcoral']
    explode = (0, 0.1)  # Explode the 2nd slice (multiple actions)
    
    # Plot pie chart
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140, textprops={'fontsize': 12})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Predictions with Single vs Multiple Actions', fontsize=14)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiple_actions_pie.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'multiple_actions_pie.pdf'))
    plt.close()

def plot_confusion_matrix(metrics, output_dir):
    """Create a heatmap visualization of the action distribution."""
    plt.figure(figsize=(12, 8))
    
    # Extract distribution data
    gt_dist = metrics.get('ground_truth_distribution', {})
    pred_dist = metrics.get('prediction_distribution', {})
    
    # Get all unique actions
    all_actions = list(set(list(gt_dist.keys()) + list(pred_dist.keys())))
    all_actions = [a for a in all_actions if a != 'unknown']
    
    # Standard order if available
    standard_order = ['check', 'call', 'fold', 'bet', 'raise']
    all_actions = [a for a in standard_order if a in all_actions] + [a for a in all_actions if a not in standard_order]
    
    # Create labels
    pred_labels = [f'Predicted: {a}' for a in all_actions] + ['Predicted: unknown']
    gt_labels = [f'Ground Truth: {a}' for a in all_actions]
    
    # Create the distribution matrix
    matrix = []
    for gt_action in all_actions:
        row = []
        gt_count = gt_dist.get(gt_action, 0)
        for pred_action in all_actions + ['unknown']:
            # How many times was pred_action predicted when gt_action was the ground truth?
            # This is an approximation since we don't have the full confusion matrix
            if gt_count > 0:
                if pred_action == gt_action:
                    # Use the accuracy as an approximation
                    correct = metrics['action_metrics'][gt_action]['accuracy'] * gt_count
                    row.append(correct / gt_count)
                else:
                    # For incorrect actions, distribute remaining probability
                    # This is a simplification since we don't have the actual counts
                    incorrect_total = (1 - metrics['action_metrics'][gt_action]['accuracy']) * gt_count
                    pred_incorrect_count = pred_dist.get(pred_action, 0) / metrics['total_examples']
                    row.append(pred_incorrect_count / (len(all_actions) - 1))
            else:
                row.append(0)
    
        matrix.append(row)
    
    # Create heatmap
    sns.heatmap(matrix, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=pred_labels, yticklabels=gt_labels,
                cbar_kws={'label': 'Proportion'})
    
    plt.title('Approximated Confusion Matrix for Action Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distribution.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'action_distribution.pdf'))
    plt.close()

def plot_summary_with_multiple_actions(metrics, output_dir):
    """Create a combined summary chart showing accuracy with multiple actions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot accuracy metrics on the left
    accuracy_metrics = {
        'Average\nAccuracy': metrics['average_accuracy'],
        'Action\nAccuracy': metrics['action_accuracy']
    }
    
    if 'value_accuracy' in metrics:
        accuracy_metrics['Value Accuracy\n(bet/raise)'] = metrics['value_accuracy']
    
    # Plot bar chart for accuracies
    bars = ax1.bar(accuracy_metrics.keys(), accuracy_metrics.values(), color='skyblue')
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom', fontsize=12)
    
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Overall Performance Metrics')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Plot multiple actions pie chart on the right
    multiple_actions = metrics.get('multiple_actions_count', 0)
    single_action = metrics['total_examples'] - multiple_actions
    
    # Create pie chart
    labels = ['Single Action\nPredictions', 'Multiple Actions\nPredictions']
    sizes = [single_action, multiple_actions]
    colors = ['lightgreen', 'lightcoral']
    explode = (0, 0.1)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140, textprops={'fontsize': 12})
    ax2.axis('equal')
    ax2.set_title('Single vs Multiple Action Predictions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_chart.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'summary_chart.pdf'))
    plt.close()

def main():
    """Main function to generate all visualizations."""
    parser = argparse.ArgumentParser(description='Generate visualizations for poker evaluation results.')
    parser.add_argument('--metrics', type=str, required=True,
                      help='Path to the metrics JSON file')
    parser.add_argument('--output', type=str, default='visualizations',
                      help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Load the metrics data
    metrics = load_metrics(args.metrics)
    
    # Create output directory
    output_dir = create_output_dir(args.output)
    
    # Generate visualizations
    plot_overall_accuracy(metrics, output_dir)
    plot_action_accuracy(metrics, output_dir)
    plot_bet_raise_value_accuracy(metrics, output_dir)
    plot_multiple_actions(metrics, output_dir)
    plot_confusion_matrix(metrics, output_dir)
    plot_summary_with_multiple_actions(metrics, output_dir)
    
    print(f"Visualizations created in {output_dir}")

if __name__ == "__main__":
    main() 