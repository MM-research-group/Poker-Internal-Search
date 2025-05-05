#!/usr/bin/env python3
import json
import os
import sys
import argparse
import re
import logging
from tqdm import tqdm
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the model loading function from helper_functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from helper_functions import load_model, create_sampling_params, find_local_model_path, setup_hf_env
except ImportError:
    logger.error("Failed to import helper functions. Make sure helper_functions.py is in the same directory.")
    sys.exit(1)

def load_dataset(file_path):
    """Load a dataset from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        return None

def extract_action(response_text):
    """Extract the optimal action from the LLM's response."""
    # Look for the phrase "The optimal action is: " followed by any text
    match = re.search(r"The optimal action is:\s*(\S+(?:\s+\d+)?)", response_text)
    if match:
        return match.group(1).strip()
    
    # Fallback pattern - look for a more general action phrase
    match = re.search(r"(?:optimal|best|correct|recommended|appropriate)\s+action\s+(?:is|would be):\s*(\S+(?:\s+\d+)?)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Look for direct action words near the end of the response
    action_words = ["call", "fold", "check", "raise", "bet"]
    for action in action_words:
        pattern = rf"\b({action}(?:\s+\d+)?)\b"
        matches = list(re.finditer(pattern, response_text, re.IGNORECASE))
        if matches:
            # Prefer matches near the end of the response
            return matches[-1].group(1).strip()
    
    # If no clear action is found
    logger.warning(f"Could not extract action from response: {response_text[:100]}...")
    return None

def evaluate_dataset(model, dataset, sampling_params, output_file=None):
    """
    Evaluate the model on a dataset of poker scenarios.
    
    Args:
        model: The loaded LLM model
        dataset: List of dictionaries with 'input' and 'optimal_action' keys
        sampling_params: Parameters for text generation
        output_file: Path to save results (optional)
    
    Returns:
        dict: Evaluation results
    """
    results = []
    correct_count = 0
    total_count = 0
    
    logger.info(f"Evaluating model on {len(dataset)} examples...")
    
    for item in tqdm(dataset, desc="Processing"):
        prompt = item["input"]
        optimal_action = item.get("optimal_action", "").strip().lower()
        
        # Skip if no expected action
        if not optimal_action:
            logger.warning(f"Example missing 'optimal_action' field, skipping")
            continue
        
        # Generate response
        try:
            outputs = model.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            
            # Extract predicted action
            predicted_action = extract_action(response)
            if predicted_action:
                predicted_action = predicted_action.lower()
                
                # Check if prediction matches expected action
                is_correct = predicted_action == optimal_action
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                # Save result
                result_item = {
                    "input": prompt,
                    "optimal_action": optimal_action,
                    "predicted_action": predicted_action,
                    "full_response": response,
                    "is_correct": is_correct
                }
                results.append(result_item)
                
                logger.debug(f"Expected: {optimal_action}, Predicted: {predicted_action}, Correct: {is_correct}")
            else:
                logger.warning(f"Failed to extract action from response")
                
        except Exception as e:
            logger.error(f"Error processing example: {e}")
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Prepare final results
    evaluation_results = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "results": results
    }
    
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    # Save results if output file is provided
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
    
    return evaluation_results

def setup_environment():
    """Set up the Hugging Face environment and validate token exists."""
    # Set multiprocessing method for vLLM
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    # Set up HF environment
    hf_home = setup_hf_env()
    
    # Check for HF_TOKEN
    if not os.getenv('HF_TOKEN'):
        logger.warning("No HF_TOKEN found in environment variables. Limited to non-gated models.")
    
    return hf_home

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a language model on poker decision making.')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help='Model name to evaluate (format: "org/model_name")')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to JSON dataset file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the evaluation results')
    parser.add_argument('--temp', type=float, default=0.1,
                       help='Temperature for sampling (lower is more deterministic)')
    parser.add_argument('--max_tokens', type=int, default=300,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--device_num', type=str, default=None,
                       help='GPU device number to use (sets CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    return parser.parse_args()

def load_model_instance(model_name):
    """Load a model by first checking if it's available locally.
    
    Args:
        model_name (str): Model name in the format "org/model_name"
        
    Returns:
        The loaded model instance
    """
    logger.info(f"Loading model: {model_name}")
    
    # Find local path for the model
    local_path = find_local_model_path(model_name)
    
    if local_path:
        logger.info(f"Using local model path: {local_path}")
        return load_model(local_path)
    else:
        logger.info(f"Model not found locally. Will download from Hugging Face Hub.")
        return load_model(model_name)

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set CUDA device if specified
    if args.device_num is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num
        logger.info(f"Using GPU device {args.device_num}")
    
    # Set up environment (including HF_TOKEN)
    setup_environment()
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error(f"Failed to load dataset from {args.dataset}")
        return
    
    # Load model
    model = None
    try:
        model = load_model_instance(args.model)
        logger.info(f"Successfully loaded model: {args.model}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Create sampling parameters
    sampling_params = create_sampling_params(
        temperature=args.temp,
        max_tokens=args.max_tokens
    )
    
    # Evaluate
    try:
        results = evaluate_dataset(
            model=model,
            dataset=dataset,
            sampling_params=sampling_params,
            output_file=args.output
        )
        
        # Print summary
        logger.info(f"Final accuracy: {results['accuracy']:.2%}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
    finally:
        # Clean up resources
        cleanup_resources(model)

def cleanup_resources(model):
    """Clean up resources to prevent NCCL warnings and memory leaks."""
    # Properly destroy process groups
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            logger.info("Destroying process group...")
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Error during process group cleanup: {e}")
    
    # Free model resources
    if model is not None:
        try:
            logger.info("Freeing model resources...")
            del model
        except Exception as e:
            logger.warning(f"Error freeing model resources: {e}")
    
    # Clear CUDA cache
    try:
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared")
    except Exception as e:
        logger.warning(f"Error clearing CUDA cache: {e}")

if __name__ == "__main__":
    main() 

    '''
    Expected output:
        {
            "accuracy": 0.0,
            "correct_count": 0,
            "total_count": 2,
            "results": [
                {
                    "input": "Poker scenario...",
                    "optimal_action": "bet 3",
                    "predicted_action": "call",
                    "full_response": blablabla,
                    "is_correct": false
                }
            ]
        }
    '''