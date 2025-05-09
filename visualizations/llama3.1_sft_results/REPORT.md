# Llama 3.1-8B SFT Evaluation Results Report

## Overview

This report summarizes the evaluation results for the fine-tuned Llama 3.1-8B model on the poker action prediction task. The model was tested on 10,000 examples to assess its ability to predict poker actions correctly after supervised fine-tuning (SFT).

## Key Findings

- **Overall Performance**:
  - Average accuracy: 38.58% (↑ 21.31% from baseline)
  - Action accuracy: 42.15% (↑ 24.20% from baseline)
  - Value accuracy (for bet/raise): 2.36% (↑ 2.08% from baseline)

- **Action-Specific Performance**:
  - Check: 49.82% accuracy (2,500 examples) (↑ 25.42% from baseline)
  - Call: 50.72% accuracy (2,500 examples) (↑ 24.64% from baseline)
  - Fold: 38.24% accuracy (2,500 examples) (↑ 24.00% from baseline)
  - Bet: 32.21% accuracy, 5.26% value accuracy (950 examples) (↑ 20.42% action, ↑ 4.63% value)
  - Raise: 23.23% accuracy, 1.03% value accuracy (1,550 examples) (↑ 19.03% action, ↑ 0.97% value)

- **Multiple Actions Issue**:
  - **Only 8.75%** of predictions contained multiple actions (↓ 44.21% from baseline)
  - This dramatic improvement demonstrates successful format learning from SFT

## Analysis

1. **Substantial Overall Performance Improvement**: The fine-tuned model shows a dramatic improvement in overall accuracy metrics (more than doubling from baseline), demonstrating that SFT is highly effective for this task.

2. **Action Type Performance Gains**: 
   - Performance substantially improved across all action types
   - "Call" and "check" actions now approach 50% accuracy
   - Simpler actions (check, call, fold) show the strongest performance
   - Complex actions (bet, raise) show the most dramatic relative improvement

3. **Value Prediction Improvements**: 
   - Value accuracy for bet actions improved significantly (over 8x increase from baseline)
   - Raise value prediction still challenging but shows meaningful improvement

4. **Dramatic Format Learning**: 
   - The percentage of predictions with multiple actions decreased from 52.96% to just 8.75%
   - This 44% absolute reduction demonstrates the model has successfully learned to generate properly formatted single-action outputs

## Distribution Analysis

- **Ground Truth Distribution**:
  - Same as baseline: well balanced with 2,500 examples each for check, call, and fold
  - 950 bet examples and 1,550 raise examples

- **Prediction Distribution**:
  - Only 875 predictions with multiple actions (labeled as "null") (↓ 4,421 from baseline)
  - 1,854 "unknown" predictions (↓ 1,055 from baseline)
  - Dramatically improved distribution of predictions:
    - Call: 2,644 (↑ 1,992 from baseline)
    - Check: 2,117 (↑ 1,507 from baseline)
    - Fold: 1,212 (↑ 856 from baseline)
    - Bet: 678 (↑ 566 from baseline)
    - Raise: 620 (↑ 555 from baseline)

## Recommendations

1. **Further Specialized Value Training**: While action type prediction and format are now reasonably strong, value prediction remains an area for improvement, particularly for raise actions.

2. **Context Refinement**: Given the success in format learning, focus now on improving the model's poker strategy understanding by refining game state representations.

3. **Action Balance Fine-Tuning**: The model has improved but still shows some bias toward call/check actions. Additional targeted fine-tuning may further improve balance.

4. **Human Evaluation**: With this substantial improvement, human evaluation of the model's strategic choices would be valuable to assess poker skill beyond format correctness.

5. **Larger Model Exploration**: Consider testing if these SFT techniques transfer effectively to larger models for even better performance.

## Comparison to Baseline

| Metric | Baseline | SFT Model | Improvement |
|--------|----------|-----------|-------------|
| Average Accuracy | 17.27% | 38.58% | +21.31% |
| Action Accuracy | 17.95% | 42.15% | +24.20% |
| Value Accuracy | 0.28% | 2.36% | +2.08% |
| Multiple Actions | 52.96% | 8.75% | -44.21% |

## Visualizations

The following visualizations are available in this directory:
- overall_accuracy.png/pdf: Bar chart of overall accuracy metrics
- action_accuracy.png/pdf: Accuracy breakdown by action type
- bet_raise_value_accuracy.png/pdf: Comparison of action vs. value accuracy for bet/raise
- multiple_actions_pie.png/pdf: Proportion of single vs. multiple action predictions
- action_distribution.png/pdf: Approximated confusion matrix for actions
- summary_chart.png/pdf: Combined chart of key metrics 