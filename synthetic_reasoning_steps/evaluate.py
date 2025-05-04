import json
import os
import argparse
import time
from typing import Dict, List, Any, Tuple
import torch
from tqdm import tqdm

from helper_functions import load_model, initialize_tokenizer

def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load test data from a JSON file.
    
    Args:
        data_path: Path to the test data JSON file
        
    Returns:
        List of test examples as dictionaries
    """
    print(f"Loading test data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def process_example(example: Dict[str, Any], model, model_name: str) -> Tuple[str, bool]:
    """
    Process a single example and get model prediction.
    
    Args:
        example: A dictionary containing the example data
        model: The loaded model
        model_name: Name of the model being used
        
    Returns:
        Tuple of (prediction, is_correct)
    """
    # Extract input and ground truth
    instruction = example["instruction"]
    ground_truth = example["output"]
    
    # Get model prediction (implementation will depend on model type)
    # This is just a placeholder - actual implementation needed
    if "gpt" in model_name.lower():
        # For OpenAI models
        prediction = "placeholder"  # TODO: Implement OpenAI API call
    else:
        # For local models via vllm
        prediction = "placeholder"  # TODO: Implement vllm inference
        
    # Compare prediction with ground truth (simple exact match)
    is_correct = prediction.strip().lower() == ground_truth.strip().lower()
    
    return prediction, is_correct

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate evaluation metrics based on results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary containing metrics
    """
    correct_count = sum(1 for result in results if result["is_correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Add more metrics as needed
    metrics = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count
    }
    
    return metrics

def save_results(results: List[Dict[str, Any]], metrics: Dict[str, float], 
                output_dir: str, model_name: str) -> str:
    """
    Save evaluation results to a file.
    
    Args:
        results: List of result dictionaries
        metrics: Dictionary of calculated metrics
        output_dir: Directory to save results
        model_name: Name of the model being evaluated
        
    Returns:
        Path to the saved results file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create safe model name for file
    safe_model_name = model_name.replace("/", "_")
    
    # Create results object with metrics and individual examples
    output = {
        "model_name": model_name,
        "metrics": metrics,
        "results": results
    }
    
    # Save to file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(output_dir, f"{safe_model_name}_eval_{timestamp}.json")
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
        
    print(f"Results saved to: {output_path}")
    return output_path

def evaluate_model(model_name: str, test_data_path: str, output_dir: str, 
                  batch_size: int = 1) -> Dict[str, float]:
    """
    Evaluate a model on a test dataset.
    
    Args:
        model_name: Name of the model to evaluate
        test_data_path: Path to the test data
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load model
    print(f"Loading model: {model_name}")
    model = load_model(model_name)
    
    # Load test data
    test_data = load_test_data(test_data_path)
    
    # Process examples and collect results
    results = []
    print(f"Evaluating model on {len(test_data)} examples...")
    
    for example in tqdm(test_data):
        prediction, is_correct = process_example(example, model, model_name)
        
        result = {
            "instruction": example["instruction"],
            "ground_truth": example["output"],
            "prediction": prediction,
            "is_correct": is_correct
        }
        results.append(result)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    print(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
    
    # Save results
    save_results(results, metrics, output_dir, model_name)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on poker decision tasks")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--data", type=str, required=True, help="Path to test data JSON file")
    parser.add_argument("--output_dir", type=str, default="synthetic_reasoning_steps/evaluation_results", 
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_name=args.model,
        test_data_path=args.data,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
