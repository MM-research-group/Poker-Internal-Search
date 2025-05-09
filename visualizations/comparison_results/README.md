# Comparative Analysis: Baseline vs SFT Model

This directory contains side-by-side visualizations comparing the baseline Llama 3.1-8B model with the fine-tuned (SFT) version on the poker action prediction task.

## Visualization Types

1. **Overall Accuracy Comparison** (`comparison_overall_accuracy.png/pdf`)
   - Side-by-side comparison of average accuracy, action accuracy, and value accuracy
   - Shows absolute improvement from baseline to SFT

2. **Action-Specific Accuracy Comparison** (`comparison_action_accuracy.png/pdf`)
   - Compares accuracy for each action type between baseline and SFT models
   - Includes action counts and absolute improvement for each action type

3. **Bet/Raise Value Accuracy Comparison** (`comparison_bet_raise_value.png/pdf`)
   - Detailed comparison of action vs. value accuracy for bet and raise actions
   - Highlights the greater challenge of value prediction vs. action prediction

4. **Multiple Actions Analysis** (`comparison_multiple_actions.png/pdf`)
   - Comparison of single vs. multiple action predictions between models
   - Shows the dramatic reduction in multiple actions after SFT (from ~53% to ~9%)

5. **Prediction Distribution Comparison** (`comparison_prediction_distribution.png/pdf`)
   - Comparative distribution of model predictions across action types
   - Demonstrates improved balance in predictions after SFT

6. **Summary Table** (`comparison_summary_table.png/pdf`)
   - Table form of key metrics side-by-side with improvement indicators
   - Provides a quick reference for important performance changes

## Key Findings

1. **Format Learning**: The most dramatic improvement is in multiple action reduction (44% absolute reduction), indicating successful format learning through SFT.

2. **Overall Performance**: The SFT model achieves substantially higher accuracy across all metrics:
   - Average accuracy: 38.58% (↑ 21.31%)
   - Action accuracy: 42.15% (↑ 24.20%)
   - Value accuracy: 2.36% (↑ 2.08%)

3. **Action-Specific Gains**: All action types show substantial improvement, with check and call actions approaching 50% accuracy.

4. **Prediction Distribution**: The SFT model produces a more balanced distribution of predictions across action types, with significantly fewer "unknown" and multiple action outputs.

5. **Value Prediction Challenge**: Despite improvements, value prediction for bet/raise actions remains challenging, especially for raise actions.

## Data Sources

- Baseline metrics: `synthetic_reasoning_steps/eval_results/baseline_eval_llama3.1-8b-instruct_metrics.json`
- SFT metrics: `test_sft_eval_results/sft_eval_llama3.1-8b-instruct_metrics.json`

## Visualization Generation

These visualizations were generated using the `compare_visualizations.py` script with the following command:

```
python compare_visualizations.py --baseline synthetic_reasoning_steps/eval_results/baseline_eval_llama3.1-8b-instruct_metrics.json --sft test_sft_eval_results/sft_eval_llama3.1-8b-instruct_metrics.json --output visualizations/comparison_results
``` 