# Llama 3.1-8B SFT Evaluation Results Report

## Overview

This report summarizes the evaluation results for the fine-tuned Llama 3.1-8B model on the poker action prediction task. The model was tested on 10,000 examples to assess its ability to predict poker actions correctly after supervised fine-tuning (SFT).

## Key Findings

- **Overall Performance**:
  - Average accuracy: 23.54% (↑ 6.27% from baseline)
  - Action accuracy: 24.15% (↑ 6.20% from baseline)
  - Value accuracy (for bet/raise): 0.86% (↑ 0.58% from baseline)

- **Action-Specific Performance**:
  - Check: 30.68% accuracy (2,500 examples) (↑ 6.28% from baseline)
  - Call: 32.56% accuracy (2,500 examples) (↑ 6.48% from baseline)
  - Fold: 18.12% accuracy (2,500 examples) (↑ 3.88% from baseline)
  - Bet: 15.89% accuracy, 1.89% value accuracy (950 examples) (↑ 4.10% action, ↑ 1.26% value)
  - Raise: 7.48% accuracy, 0.32% value accuracy (1,550 examples) (↑ 3.29% action, ↑ 0.26% value)

- **Multiple Actions Issue**:
  - 43.92% of predictions contained multiple actions (↓ 9.04% from baseline)
  - These were counted as incorrect predictions

## Analysis

1. **Improved Overall Performance**: The fine-tuned model shows approximately 6% improvement in overall accuracy metrics, indicating that SFT is effective for this task, though performance remains suboptimal.

2. **Action Type Performance Gains**: 
   - Performance improved across all action types
   - "Call" and "check" actions saw the largest absolute improvements
   - Significant relative improvements for bet/raise actions, though from a low base

3. **Value Prediction Improvements**: 
   - Value accuracy for bet actions improved notably (3x increase)
   - Raise value prediction remains extremely challenging

4. **Reduced Multiple Actions Problem**: 
   - The percentage of predictions with multiple actions decreased by 9%
   - This suggests the fine-tuning helped the model learn to produce cleaner, single-action outputs

## Distribution Analysis

- **Ground Truth Distribution**:
  - Same as baseline: well balanced with 2,500 examples each for check, call, and fold
  - 950 bet examples and 1,550 raise examples

- **Prediction Distribution**:
  - 4,392 predictions with multiple actions (labeled as "null") (↓ 904 from baseline)
  - 2,473 "unknown" predictions (↓ 436 from baseline)
  - Increased predictions across all action types:
    - Call: 814 (↑ 162 from baseline)
    - Check: 767 (↑ 157 from baseline)
    - Fold: 453 (↑ 97 from baseline)
    - Bet: 151 (↑ 39 from baseline)
    - Raise: 116 (↑ 51 from baseline)

## Recommendations

1. **Enhanced Action Filtering**: While improved, multiple actions remain an issue. Consider more explicit prompt engineering or post-processing.

2. **Focused Value Training**: The model made progress on bet value prediction, but raise values remain problematic. A specialized training approach for raise values could help.

3. **Action Balancing**: The model still shows bias toward simpler actions (call, check). Consider weighted training to emphasize bet/raise actions.

4. **Input Context Engineering**: Review the game state representation provided to the model to ensure sufficient information for making complex decisions.

5. **Iterative Fine-Tuning**: These results suggest iterative fine-tuning may yield further gains, particularly if focused on areas of weakness.

## Comparison to Baseline

| Metric | Baseline | SFT Model | Improvement |
|--------|----------|-----------|-------------|
| Average Accuracy | 17.27% | 23.54% | +6.27% |
| Action Accuracy | 17.95% | 24.15% | +6.20% |
| Value Accuracy | 0.28% | 0.86% | +0.58% |
| Multiple Actions | 52.96% | 43.92% | -9.04% |

## Visualizations

The following visualizations are available in this directory:
- overall_accuracy.png/pdf: Bar chart of overall accuracy metrics
- action_accuracy.png/pdf: Accuracy breakdown by action type
- bet_raise_value_accuracy.png/pdf: Comparison of action vs. value accuracy for bet/raise
- multiple_actions_pie.png/pdf: Proportion of single vs. multiple action predictions
- action_distribution.png/pdf: Approximated confusion matrix for actions
- summary_chart.png/pdf: Combined chart of key metrics 