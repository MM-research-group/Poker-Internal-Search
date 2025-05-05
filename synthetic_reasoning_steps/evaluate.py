'''
Usage:
    python synthetic_reasoning_steps/evaluate.py [--model MODEL_NAME] [--dataset DATASET_PATH] [--output OUTPUT_PATH] [--device_num DEVICE_NUM]
    
This script processes a dataset using a language model and saves the results.
'''

import sys
import os
import argparse
import json
import logging
import time
import signal
import atexit
import torch.distributed as dist
from tqdm import tqdm
from dotenv import load_dotenv

# Import helper functions
from helper_functions import (
    load_model, setup_hf_env, find_local_model_path, 
    load_dataset, create_sampling_params, format_prompt, process_output
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
        logger.error("No HF_TOKEN found in environment. Cannot access gated models.")
        sys.exit(1)
    
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
    
    # Find local path for the model
    local_path = find_local_model_path(model_name)
    
    if local_path:
        _model = load_model(local_path)
        return _model
    else:
        logger.info(f"Model not found locally. Will download from Hugging Face Hub.")
        _model = load_model(model_name)
        return _model

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
    
    return result

def evaluate_dataset(model, dataset, prompt_template, sampling_params, output_path=None, batch_size=1):
    """Evaluate an entire dataset using the model.
    
    Args:
        model: The loaded model
        dataset (list): List of examples
        prompt_template (str): Template for formatting prompts
        sampling_params: Parameters for sampling
        output_path (str, optional): Path to save results
        batch_size (int, optional): Number of examples to process in one batch
        
    Returns:
        list: Results for each example
    """
    results = []
    
    logger.info(f"Processing {len(dataset)} examples...")
    
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
    try:
        # Process each example with a progress bar
        for i, example in enumerate(tqdm(dataset, desc="Processing dataset")):
            result = evaluate_single_example(model, example, prompt_template, sampling_params)
            results.append(result)
            
            # Save intermediate results every 100 examples
            if output_path and (i + 1) % 100 == 0:
                save_count += 1
                save_intermediate_results(results, output_path, save_count)
                
            # Optionally display some information
            logger.debug(f"Input: {result.get('prompt', '')[:100]}...")
            logger.debug(f"Output: {result.get('model_output', '')[:100]}...")
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
    
    # Save final results if output path is provided
    if output_path:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a language model on a dataset.')
    parser.add_argument('--model', type=str, default="meta-llama/meta-llama-3.1-8b-instruct",
                       help='Model name to evaluate (format: "org/model_name")')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to JSON dataset file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the evaluation results')
    parser.add_argument('--prompt_template', type=str, default=None,
                       help='Template for formatting prompts, with {field} placeholders')
    parser.add_argument('--prompt', type=str, default="Please explain what neural networks are in 3 sentences.",
                       help='Test prompt if no dataset is provided')
    parser.add_argument('--temp', type=float, default=0.7,
                       help='Temperature for sampling')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling parameter')
    parser.add_argument('--device_num', type=str, default=None,
                       help='GPU device number to use (sets CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Number of examples to process in one batch')
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
            results = evaluate_dataset(
                model=model,
                dataset=dataset,
                prompt_template=args.prompt_template,
                sampling_params=sampling_params,
                output_path=args.output,
                batch_size=args.batch_size
            )
            
            logger.info(f"Evaluation completed for {len(results)} examples")
        else:
            # Create a single example from the prompt
            example = {"input": args.prompt}
            
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
            logger.info(result["model_output"])
            logger.info("-" * 50)
            logger.info(f"Generation time: {result['generation_time']:.2f} seconds")
        
        logger.info("Evaluation completed successfully!")
    finally:
        # Make sure resources are cleaned up even if there's an error
        cleanup_resources()

if __name__ == "__main__":
    main()
