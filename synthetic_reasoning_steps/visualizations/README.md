# Poker Model Evaluation Visualizations

This directory contains visualizations for the poker model evaluation results.

## Visualization Types

The script generates the following visualizations:

1. **Overall Accuracy Metrics** (`overall_accuracy.png/pdf`)
   - Bar chart showing average accuracy, action accuracy, and value accuracy (for bet/raise actions)

2. **Action-Specific Accuracy** (`action_accuracy.png/pdf`) 
   - Bar chart showing accuracy for each action type (check, call, fold, bet, raise)
   - Includes the count of examples for each action type

3. **Bet/Raise Value Accuracy** (`bet_raise_value_accuracy.png/pdf`)
   - Grouped bar chart comparing action accuracy vs. value accuracy for bet and raise actions

4. **Multiple Actions Analysis** (`multiple_actions_pie.png/pdf`)
   - Pie chart showing the proportion of predictions with single vs. multiple actions

5. **Action Distribution (Confusion Matrix)** (`action_distribution.png/pdf`)
   - Heatmap approximating a confusion matrix for action predictions

6. **Summary Chart** (`summary_chart.png/pdf`)
   - Combined visualization showing overall metrics alongside multiple actions analysis

## How to Generate Visualizations

To generate the visualizations, run the following command from the project root:

```bash
python synthetic_reasoning_steps/visualizations/generate_plots.py --metrics path/to/metrics.json --output path/to/save/visualizations
```

Where:
- `--metrics` is the path to your evaluation metrics JSON file
- `--output` (optional) is the directory where visualizations will be saved

Example:
```bash
python synthetic_reasoning_steps/visualizations/generate_plots.py --metrics eval_results/baseline_eval_llama3.1-8b_metrics.json --output synthetic_reasoning_steps/visualizations/llama31_results
```

## Requirements

The script requires the following Python packages:
- matplotlib
- seaborn
- numpy

You can install them with:
```bash
pip install matplotlib seaborn numpy
```

## Usage in Research Papers

These visualizations are designed to be publication-ready for inclusion in research papers. They are saved in both PNG (for web/previews) and PDF (for publication) formats. The PDF versions maintain vector graphics quality for publication purposes. 