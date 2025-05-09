'''
Usage:
    python synthetic_reasoning_steps/baseline_evaluate.py [--model MODEL_NAME] [--dataset DATASET_PATH] [--output OUTPUT_PATH] [--device_num DEVICE_NUM]
    
This script processes a poker decision dataset using a language model, compares predictions with ground truth,
and calculates evaluation metrics.
'''

import sys
import os
import argparse
import json
import logging
import time
import signal
import atexit
import re
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from collections import Counter

# Import helper functions
from helper_functions import (
    load_model, setup_hf_env, find_local_model_path, 
    load_dataset, create_sampling_params, format_prompt, process_output,
    process_batch_examples
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
_model = None

def cleanup_resources():
    """Clean up resources to avoid warnings and memory leaks."""
    global _model
    
    # Properly destroy process groups
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.debug("Successfully destroyed process group")
    except Exception as e:
        logger.debug(f"Error while destroying process group: {e}")
    
    # Free model resources
    if _model is not None:
        try:
            del _model
            _model = None
            logger.debug("Model resources freed")
        except Exception as e:
            logger.debug(f"Error while freeing model resources: {e}")
    
    # Clear CUDA cache
    try:
        import torch
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")
    except Exception as e:
        logger.debug(f"Error while clearing CUDA cache: {e}")

# Register cleanup function to be called at exit
atexit.register(cleanup_resources)

# Register signal handlers
def signal_handler(sig, frame):
    """Handle signals gracefully by cleaning up resources before exit."""
    logger.info(f"Received signal {sig}, cleaning up resources...")
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def setup_environment():
    """Set up the Hugging Face environment and validate token exists."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Set multiprocessing method for vLLM
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    # Check for HF_TOKEN
    hf_home = setup_hf_env()
    if not os.getenv('HF_TOKEN'):
        logger.warning("No HF_TOKEN found in environment. May not be able to access gated models.")
    
    return hf_home

def load_model_instance(model_name):
    """Load a model by first checking if it's available locally.
    
    Args:
        model_name (str): Model name in the format "org/model_name"
        
    Returns:
        The loaded model instance
    """
    global _model
    
    logger.info(f"Loading model: {model_name}")
    
    # Override to use the specific local path for Llama 3.1-8B-Instruct
    if "meta-llama/meta-llama-3.1-8b-instruct" in model_name.lower():
        local_path = "/srv/share/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
        logger.info(f"Using local Llama 3.1-8B-Instruct model at: {local_path}")
        _model = load_model(local_path)
        return _model
    
    # Find local path for other models
    local_path = find_local_model_path(model_name)
    
    if local_path:
        _model = load_model(local_path)
        return _model
    else:
        logger.error(f"Model not found locally: {model_name}")
        logger.error("This script requires local models - please ensure the model exists at /srv/share/huggingface/hub/")
        sys.exit(1)

def evaluate_single_example(model, example, prompt_template, sampling_params):
    """Evaluate a single example with the model.
    
    Args:
        model: The loaded model
        example (dict): Example data
        prompt_template (str): Template for formatting prompts
        sampling_params: Parameters for sampling
        
    Returns:
        dict: Original example with model output added
    """
    # Format the prompt for this example
    prompt = format_prompt(example, prompt_template)
    
    # Generate the response
    start_time = time.time()
    outputs = model.generate([prompt], sampling_params)
    end_time = time.time()
    
    # Process the output
    output_text = process_output(outputs[0])
    
    # Add results to the example
    result = example.copy()
    result["model_output"] = output_text
    result["prompt"] = prompt
    result["generation_time"] = end_time - start_time
    
    # Evaluate the prediction against ground truth
    eval_result = evaluate_poker_prediction(result["output"], output_text)
    result.update(eval_result)
    
    return result

def evaluate_poker_prediction(ground_truth, prediction):
    """Evaluate poker action prediction against ground truth.
    
    Args:
        ground_truth (str): Ground truth poker action
        prediction (str): Predicted poker action
        
    Returns:
        dict: Evaluation results including accuracy and details
    """
    # Normalize by converting to lowercase and stripping whitespace
    gt = ground_truth.lower().strip()
    pred = prediction.lower().strip()
    
    # Initialize result dictionary
    result = {
        "accuracy": 0.0,
        "action_correct": False,
        "value_correct": False,
        "matched_action": None,
        "contains_multiple_actions": False
    }
    
    # Check if prediction contains multiple actions
    simple_actions = ["check", "fold", "call"]
    action_count = 0
    for action in simple_actions:
        if action in pred:
            action_count += 1
    
    if "bet" in pred:
        action_count += 1
    
    if "raise" in pred:
        action_count += 1
    
    # Mark if multiple actions detected
    if action_count > 1:
        result["contains_multiple_actions"] = True
        result["accuracy"] = 0.0
        return result
    
    # For simple actions (check, fold, call)
    if gt in simple_actions:
        result["matched_action"] = gt if gt in pred else "unknown"
        result["action_correct"] = gt in pred
        result["accuracy"] = 1.0 if result["action_correct"] else 0.0
        return result
    
    # For bet/raise with values
    bet_match_gt = re.match(r"bet\s+(\d+)", gt)
    raise_match_gt = re.match(r"raise\s+(\d+)", gt)
    
    if bet_match_gt:
        gt_action = "bet"
        gt_value = int(bet_match_gt.group(1))
        result["matched_action"] = "bet" if "bet" in pred else "unknown"
        result["action_correct"] = "bet" in pred
        
        if result["action_correct"]:
            result["accuracy"] = 0.6  # 60% for getting the action right
            
            # Try to extract value
            bet_match_pred = re.search(r"bet\s+(\d+)", pred)
            if bet_match_pred:
                pred_value = int(bet_match_pred.group(1))
                if pred_value == gt_value:
                    result["value_correct"] = True
                    result["accuracy"] = 1.0
        
        return result
    
    if raise_match_gt:
        gt_action = "raise"
        gt_value = int(raise_match_gt.group(1))
        result["matched_action"] = "raise" if "raise" in pred else "unknown"
        result["action_correct"] = "raise" in pred
        
        if result["action_correct"]:
            result["accuracy"] = 0.6  # 60% for getting the action right
            
            # Try to extract value
            raise_match_pred = re.search(r"raise\s+(\d+)", pred)
            if raise_match_pred:
                pred_value = int(raise_match_pred.group(1))
                if pred_value == gt_value:
                    result["value_correct"] = True
                    result["accuracy"] = 1.0
        
        return result
    
    # If we get here, ground truth format is unexpected
    result["matched_action"] = "unknown"
    return result

def parse_poker_action(action_text):
    """Parse a poker action into action type and value (if applicable).
    
    Args:
        action_text (str): Text containing poker action
        
    Returns:
        tuple: (action_type, value) - action_type is a string, value is int or None
    """
    # Simple actions
    if "check" in action_text:
        return "check", None
    if "fold" in action_text:
        return "fold", None
    if "call" in action_text:
        return "call", None
    
    # Bet with value
    bet_match = re.search(r"bet\s+(\d+)", action_text)
    if bet_match:
        return "bet", int(bet_match.group(1))
    
    # Raise with value
    raise_match = re.search(r"raise\s+(\d+)", action_text)
    if raise_match:
        return "raise", int(raise_match.group(1))
    
    # If the action contains bet or raise but couldn't extract a value
    if "bet" in action_text:
        return "bet", None
    if "raise" in action_text:
        return "raise", None
    
    # If no match, return unknown
    return "unknown", None

def calculate_metrics(results):
    """Calculate evaluation metrics for the entire dataset.
    
    Args:
        results (list): List of results for each example
        
    Returns:
        dict: Calculated metrics
    """
    if not results:
        return {"error": "No results to calculate metrics"}
    
    # Initialize counters and accumulators
    total = len(results)
    total_accuracy = 0.0
    action_correct_count = 0
    value_correct_count = 0
    multiple_actions_count = 0
    action_counts = Counter()
    action_correct_counts = Counter()
    value_correct_counts = Counter()
    
    # Count examples by action type
    gt_action_counts = Counter()
    pred_action_counts = Counter()
    
    for result in results:
        # Skip examples without evaluation results
        if "accuracy" not in result:
            continue
        
        # Track multiple actions
        if result.get("contains_multiple_actions", False):
            multiple_actions_count += 1
        
        # Extract ground truth action
        gt_action, _ = parse_poker_action(result["output"].lower().strip())
        gt_action_counts[gt_action] += 1
        
        # Extract predicted action
        pred_action = result.get("matched_action", "unknown")
        pred_action_counts[pred_action] += 1
        
        # Accumulate accuracy
        total_accuracy += result["accuracy"]
        
        # Count correct actions
        if result["action_correct"]:
            action_correct_count += 1
            action_correct_counts[gt_action] += 1
            
            # Count correct values for bet/raise actions
            if gt_action in ["bet", "raise"] and result.get("value_correct", False):
                value_correct_count += 1
                value_correct_counts[gt_action] += 1
        
        # Track action types
        action_counts[gt_action] += 1
    
    # Calculate overall metrics
    metrics = {
        "total_examples": total,
        "average_accuracy": total_accuracy / total if total > 0 else 0,
        "action_accuracy": action_correct_count / total if total > 0 else 0,
        "multiple_actions_percentage": multiple_actions_count / total if total > 0 else 0,
    }
    
    # Calculate action-specific metrics
    action_metrics = {}
    for action in action_counts:
        action_count = action_counts[action]
        action_metrics[action] = {
            "count": action_count,
            "percentage": action_count / total if total > 0 else 0,
            "accuracy": action_correct_counts[action] / action_count if action_count > 0 else 0,
        }
        
        # Add value accuracy for bet/raise actions
        if action in ["bet", "raise"]:
            action_metrics[action]["value_accuracy"] = (
                value_correct_counts[action] / action_count if action_count > 0 else 0
            )
            
            # Update overall value accuracy metrics
            bet_raise_count = action_counts["bet"] + action_counts["raise"]
            metrics["value_accuracy"] = (
                value_correct_count / bet_raise_count if bet_raise_count > 0 else 0
            )
    
    metrics["action_metrics"] = action_metrics
    
    # Add confusion matrix data
    metrics["ground_truth_distribution"] = {k: v for k, v in gt_action_counts.items()}
    metrics["prediction_distribution"] = {k: v for k, v in pred_action_counts.items()}
    metrics["multiple_actions_count"] = multiple_actions_count
    
    return metrics

def evaluate_dataset(model, dataset, prompt_template, sampling_params, output_path=None, batch_size=8):
    """Evaluate an entire dataset using the model with batch processing.
    
    Args:
        model: The loaded model
        dataset (list): List of examples
        prompt_template (str): Template for formatting prompts
        sampling_params: Parameters for sampling
        output_path (str, optional): Path to save results
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (results, metrics) - results for each example and calculated metrics
    """
    results = []
    
    logger.info(f"Processing {len(dataset)} examples with batch size {batch_size}...")
    
    # Save results periodically to avoid losing all progress if something fails
    def save_intermediate_results(results_so_far, output_path, save_count):
        """Save intermediate results to avoid losing progress."""
        if output_path and results_so_far:
            interim_path = f"{os.path.splitext(output_path)[0]}_interim_{save_count}{os.path.splitext(output_path)[1]}"
            try:
                os.makedirs(os.path.dirname(interim_path), exist_ok=True)
                with open(interim_path, 'w', encoding='utf-8') as f:
                    json.dump(results_so_far, f, ensure_ascii=False, indent=2)
                logger.debug(f"Intermediate results saved to {interim_path}")
            except Exception as e:
                logger.warning(f"Error saving intermediate results: {e}")
    
    # Process examples in batches
    save_count = 0
    
    # Split dataset into larger chunks for intermediate saving
    chunk_size = 100  # Save after each 100 examples
    dataset_chunks = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]
    
    try:
        # Process each chunk
        for chunk_idx, chunk in enumerate(tqdm(dataset_chunks, desc="Processing dataset chunks")):
            # Process this chunk in batches
            chunk_results = []
            
            # Further split the chunk into batches
            batch_chunks = [chunk[i:i+batch_size] for i in range(0, len(chunk), batch_size)]
            
            # Process each batch in the chunk
            for batch in tqdm(batch_chunks, desc=f"Chunk {chunk_idx+1}/{len(dataset_chunks)}", leave=False):
                # Process batch using the helper function
                batch_results = process_batch_examples(
                    model=model,
                    examples=batch,
                    prompt_template=prompt_template,
                    sampling_params=sampling_params,
                    batch_size=len(batch)  # Use actual batch size
                )
                
                # Evaluate each result in the batch
                for result in batch_results:
                    # Evaluate the prediction against ground truth
                    eval_result = evaluate_poker_prediction(result["output"], result["model_output"])
                    result.update(eval_result)
                    chunk_results.append(result)
                
            # Add results from this chunk to overall results
            results.extend(chunk_results)
            
            # Save intermediate results after each chunk
            save_count += 1
            save_intermediate_results(results, output_path, save_count)
            
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user. Saving partial results...")
        if output_path:
            save_intermediate_results(results, output_path, "interrupted")
        raise
    except Exception as e:
        logger.error(f"Error during dataset processing: {e}")
        if output_path:
            save_intermediate_results(results, output_path, "error")
        raise
    
    logger.info(f"Processed {len(results)} examples")
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    logger.info("Metrics calculated")
    
    # Save final results if output path is provided
    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save detailed results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")
            
            # Save metrics separately
            metrics_path = f"{os.path.splitext(output_path)[0]}_metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
    
    return results, metrics

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a language model on a poker dataset.')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help='Model name to evaluate (format: "org/model_name")')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to JSON dataset file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the evaluation results')
    parser.add_argument('--prompt_template', type=str, default="{input}",
                       help='Template for formatting prompts, with {field} placeholders')
    parser.add_argument('--prompt', type=str, default="What poker action should I take?",
                       help='Test prompt if no dataset is provided')
    parser.add_argument('--temp', type=float, default=0.7,
                       help='Temperature for sampling')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling parameter')
    parser.add_argument('--device_num', type=str, default=None,
                       help='GPU device number to use (sets CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce logging output')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level based on quiet flag
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Set CUDA_VISIBLE_DEVICES if a device number was specified
    if args.device_num is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={args.device_num}")
    
    # Set up environment
    setup_environment()
    
    # Load model
    try:
        model = load_model_instance(args.model)
        logger.info(f"Successfully loaded model: {args.model}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Please check if the model name is correct or if the model is available locally.")
        return
    
    # Create sampling parameters
    sampling_params = create_sampling_params(
        temperature=args.temp,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )
    
    # Process dataset if provided, otherwise process single prompt
    try:
        if args.dataset:
            # Load dataset
            dataset = load_dataset(args.dataset)
            if not dataset:
                logger.error(f"Failed to load dataset or dataset is empty: {args.dataset}")
                return
            
            # Process dataset
            results, metrics = evaluate_dataset(
                model=model,
                dataset=dataset,
                prompt_template=args.prompt_template,
                sampling_params=sampling_params,
                output_path=args.output,
                batch_size=args.batch_size
            )
            
            # Display summary metrics
            logger.info("\nEvaluation Results:")
            logger.info("-" * 50)
            logger.info(f"Total examples: {metrics['total_examples']}")
            logger.info(f"Average accuracy: {metrics['average_accuracy']:.4f}")
            logger.info(f"Action accuracy: {metrics['action_accuracy']:.4f}")
            if 'value_accuracy' in metrics:
                logger.info(f"Value accuracy (for bet/raise): {metrics['value_accuracy']:.4f}")
            logger.info(f"Multiple actions detected: {metrics['multiple_actions_count']} examples ({metrics['multiple_actions_percentage']:.2%})")
            
            # Display action-specific metrics
            logger.info("\nAction-specific metrics:")
            for action, action_metrics in metrics.get('action_metrics', {}).items():
                logger.info(f"  {action}: {action_metrics['count']} examples, "
                           f"accuracy: {action_metrics['accuracy']:.4f}")
                if 'value_accuracy' in action_metrics:
                    logger.info(f"    value accuracy: {action_metrics['value_accuracy']:.4f}")
            
            logger.info("-" * 50)
            logger.info(f"Evaluation completed for {len(results)} examples")
        else:
            # Create a single example from the prompt
            example = {"input": args.prompt, "output": "fold"}  # Placeholder output for testing
            
            # Evaluate single example
            result = evaluate_single_example(
                model=model,
                example=example,
                prompt_template=args.prompt_template,
                sampling_params=sampling_params
            )
            
            # Display the result
            logger.info("\nModel output:")
            logger.info("-" * 50)
            logger.info(f"Input: {result['input']}")
            logger.info(f"Ground truth: {result['output']}")
            logger.info(f"Model prediction: {result['model_output']}")
            logger.info(f"Accuracy: {result['accuracy']:.2f}")
            logger.info(f"Action correct: {result['action_correct']}")
            if 'value_correct' in result:
                logger.info(f"Value correct: {result['value_correct']}")
            logger.info("-" * 50)
            logger.info(f"Generation time: {result['generation_time']:.2f} seconds")
        
        logger.info("Evaluation completed successfully!")
    finally:
        # Make sure resources are cleaned up even if there's an error
        cleanup_resources()

if __name__ == "__main__":
    main()
