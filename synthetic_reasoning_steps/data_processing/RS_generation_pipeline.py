# RS_generation_pipeline.py

'''
Usage:
    python synthetic_reasoning_steps/data_processing/RS_generation_pipeline.py
'''

import json
from google import genai
import sys
import os
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm
import time
from typing import List, Dict, Any, Tuple
import random

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_reasoning_prompts import (
    prompt_proposer,
    prompt_board_and_range_verifier,
    prompt_meta_verifier,
)

# Initialize Gemini client
api_key = os.environ.get("GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key)

def call_model(prompt_text, role="user", model="gemini-2.5-pro-preview-03-25", max_retries=5, initial_backoff=10):
    """
    Calls the Google Gemini API with the given prompt and returns the model response as a string.
    Implements exponential backoff for handling rate limit errors.
    
    Args:
        prompt_text: The prompt to send to the model
        role: Not used for Gemini but kept for compatibility
        model: The model to use (e.g., "gemini-2.5-pro-preview-03-25")
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        
    Returns:
        String: The model's response text
    """
    for attempt in range(max_retries + 1):  # +1 for the initial attempt
        try:
            # Send the prompt to Gemini using the updated API
            response = genai_client.models.generate_content(
                model=model,
                contents=prompt_text
            )
            
            # Return just the text content
            return response.text
                
        except Exception as e:
            # Check if it's a rate limit error 
            # (Gemini rate limit errors might have different indicators than "429")
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                if attempt < max_retries:
                    # Calculate backoff time with exponential increase and some randomness
                    backoff_time = initial_backoff * (2 ** attempt) * (0.5 + 0.5 * random.random())
                    print(f"⚠️ Rate limit reached. Retrying in {backoff_time:.1f} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff_time)
                    continue
                else:
                    print(f"❌ Maximum retries reached. Rate limit error: {e}")
            
            # For other errors, or if we've exceeded max retries
            raise e
    
    # This point should never be reached due to the raise above, but just in case:
    raise Exception("Maximum retries exceeded with no successful response")

def process_example(example, regular_model_name, meta_model_name, max_retries=5):
    """Process a single example through all four steps with retry logic"""
    try:
        gamestate = example.get("instruction", "")
        optimal_action = example.get("output", "")
        
        example_results = {
            "gamestate": gamestate,
            "optimal_action": optimal_action
        }
        
        # Step 1: Proposer Module (must be done first)
        print(f"  Running Proposer Module...")
        proposer_prompt = prompt_proposer(gamestate, optimal_action)
        
        try:
            proposer_content = call_model(proposer_prompt, model=regular_model_name, max_retries=max_retries)
        except Exception as e:
            print(f"  ❌ Proposer module failed: {e}")
            return None
        
        print(f"  ✓ Proposer complete")
        
        example_results["proposer_response"] = proposer_content
        proposed_reasoning = proposer_content
        
        # Step 2: Combined Board Analysis and Range Estimation
        print(f"  Running Combined Board Analysis & Range Estimation...")
        
        board_range_prompt = prompt_board_and_range_verifier(gamestate, optimal_action, proposed_reasoning)
        
        try:
            board_range_content = call_model(board_range_prompt, model=regular_model_name, max_retries=max_retries)
            print(f"  ✓ Combined Board Analysis & Range Estimation complete")
        except Exception as e:
            print(f"  ❌ Combined Board & Range module failed: {e}")
            return None
            
        # Store the combined output
        example_results["board_range_response"] = board_range_content
        
        # Step 3: Meta Verifier Module
        print(f"  Running Meta Verifier Module...")
        meta_prompt = prompt_meta_verifier(
            gamestate,
            optimal_action,
            proposed_reasoning,
            board_range_content,
        )
        
        try:
            meta_content = call_model(meta_prompt, model=meta_model_name, max_retries=max_retries)
            print(f"  ✓ Meta Verification complete")
        except Exception as e:
            print(f"  ❌ Meta Verifier module failed: {e}")
            # Still save partial results with error message
            meta_content = f"Error: {str(e)}"
        
        example_results["meta_response"] = meta_content
        
        return example_results
    
    except Exception as e:
        print(f"Error processing example: {e}")
        return None

def calculate_and_report_batch_time(start_time, end_time, batch_size, current_count, total_count):
    duration = end_time - start_time
    examples_per_second = batch_size / duration if duration > 0 else 0
    
    print(f"  Results saved to file ({current_count}/{total_count} examples processed)")
    print(f"  Batch completed in {duration:.2f} seconds ({examples_per_second:.2f} examples/second)")
    
    remaining_examples = total_count - current_count
    if remaining_examples > 0 and examples_per_second > 0:
        est_remaining_time = remaining_examples / examples_per_second
        hours, remainder = divmod(est_remaining_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"  Estimated time remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")

def main(data_path, batch_size=3):
    regular_model_name = "gemini-2.5-pro-preview-03-25"
    meta_model_name = "gemini-2.5-pro-preview-03-25"
    
    print(f"Loading data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    base_results_dir = "synthetic_reasoning_steps/results"
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)
    
    if "/" in meta_model_name:
        safe_model_name = meta_model_name.split("/")[-1]
    else:
        safe_model_name = meta_model_name
        
    model_results_dir = os.path.join(base_results_dir, safe_model_name)
    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir)
    
    results_file_path = os.path.join(model_results_dir, "all_results.json")
    
    # support resuming
    all_results = []
    starting_index = 0
    
    if os.path.exists(results_file_path):
        try:
            with open(results_file_path, 'r') as f:
                all_results = json.load(f)
            
            # Calculate the number of already processed examples
            starting_index = len(all_results)
            print(f"Found existing results file with {starting_index} examples already processed")
            print(f"Resuming from example {starting_index + 1}/{len(data)}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading existing results file: {e}")
            print("Starting from the beginning")
            all_results = []
            starting_index = 0
    else:
        print(f"No existing results found. Starting from the beginning.")
    
    print(f"Processing {len(data) - starting_index} examples with model: {regular_model_name}")
    print(f"Results will be saved to: {results_file_path}")
    print(f"Using batch size: {batch_size}")
    
    for batch_start in range(starting_index, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        batch = data[batch_start:batch_end]
        
        print(f"\n--- Processing batch {batch_start//batch_size + 1}, examples {batch_start+1}-{batch_end}/{len(data)} ---")
        
        batch_start_time = time.time()
        
        # Process batch in parallel
        new_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Create a dictionary to keep track of futures and their corresponding indices
            future_to_index = {
                executor.submit(process_example, example, regular_model_name, meta_model_name): i 
                for i, example in enumerate(batch)
            }
            
            # Process as futures complete
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    if result:
                        new_results.append((idx, result))
                        print(f"✓ Completed example {batch_start + idx + 1}/{len(data)}")
                except Exception as exc:
                    print(f"Example {batch_start + idx + 1} generated an exception: {exc}")
        
        new_results.sort(key=lambda x: x[0])
        for _, result in new_results:
            all_results.append(result)
            
        with open(results_file_path, "w") as f:
            json.dump(all_results, f, indent=4)
        
        batch_end_time = time.time()
        calculate_and_report_batch_time(
            batch_start_time, 
            batch_end_time, 
            len(batch),
            len(all_results), 
            len(data)
        )
    
    print("\n=================================================")
    print(f"Pipeline processing complete! All {len(all_results)}/{len(data)} examples processed.")
    print(f"Results saved to: {results_file_path}")
    print("=================================================")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "pokerbench_data", 
                            "withpotodds_postflop_60k_sampled.json")
    # Adjust batch size based on your API rate limits and system capabilities
    main(data_path, batch_size=1000)
