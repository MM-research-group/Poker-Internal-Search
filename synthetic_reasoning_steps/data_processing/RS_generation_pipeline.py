# RS_generation_pipeline.py

'''
Usage:
    python synthetic_reasoning_steps/data_processing/RS_generation_pipeline.py
'''

import json
from openai import OpenAI
import sys
import os
from dotenv import load_dotenv
import concurrent.futures
from tqdm import tqdm
import time
from typing import List, Dict, Any, Tuple

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_reasoning_prompts import (
    prompt_proposer,
    prompt_board_verifier,
    prompt_range_estimation_verifier,
    prompt_meta_verifier,
)

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def call_model(prompt_text, role="user", model="deepseek-reasoner"):
    """
    Calls the DeepSeek API with the given prompt and returns the model response as a string.
    This function expects the prompt to be provided as a string.
    
    For DeepSeek models, we can access both reasoning content and final content.
    """
    messages = [{"role": role, "content": prompt_text}]
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4000  # Conservative default
    )
    
    try:
        reasoning_content = completion.choices[0].message.reasoning_content
        content = completion.choices[0].message.content
        return reasoning_content, content
    except AttributeError:
        return None, completion.choices[0].message.content

def call_model_async(prompt_text, role="user", model="deepseek-reasoner"):
    """Async version of call_model for parallel processing"""
    return call_model(prompt_text, role, model)

def process_example(example, regular_model_name, meta_model_name):
    """Process a single example through all four steps"""
    try:
        gamestate = example.get("instruction", "")
        optimal_action = example.get("output", "")
        
        example_results = {}
        
        # Step 1: Proposer Module (must be done first)
        print(f"  Running Proposer Module...")
        proposer_prompt = prompt_proposer(gamestate, optimal_action)
        proposer_reasoning, proposer_content = call_model(proposer_prompt, model=regular_model_name)
        
        example_results["proposer_output"] = {
            "reasoning_content": proposer_reasoning,
            "final_response": proposer_content,
            "gamestate": gamestate,
            "optimal_action": optimal_action
        }
        proposed_reasoning = proposer_content
        
        # Steps 2 & 3: Run Board Analysis and Range Estimation in parallel
        print(f"  Running Board Analysis & Range Estimation in parallel...")
        
        # parallel execution
        board_prompt = prompt_board_verifier(gamestate, optimal_action, proposed_reasoning)
        range_estimation_prompt = prompt_range_estimation_verifier(gamestate, optimal_action, proposed_reasoning)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            board_future = executor.submit(call_model_async, board_prompt, "user", regular_model_name)
            range_future = executor.submit(call_model_async, range_estimation_prompt, "user", regular_model_name)
            
            board_reasoning, board_content = board_future.result()
            range_reasoning, range_content = range_future.result()
        
        example_results["board_output"] = {
            "reasoning_content": board_reasoning,
            "final_response": board_content,
            "gamestate": gamestate,
            "optimal_action": optimal_action
        }
        
        example_results["range_estimation_output"] = {
            "reasoning_content": range_reasoning,
            "final_response": range_content,
            "gamestate": gamestate,
            "optimal_action": optimal_action
        }
        
        # Step 4: Meta Verifier Module (must be done after steps 2 & 3)
        print(f"  Running Meta Verifier Module...")
        meta_prompt = prompt_meta_verifier(
            gamestate,
            optimal_action,
            proposed_reasoning,
            board_content,
            range_content
        )
        meta_reasoning, meta_content = call_model(meta_prompt, model=meta_model_name)
        
        example_results["meta_output"] = {
            "reasoning_content": meta_reasoning,
            "final_response": meta_content,
            "gamestate": gamestate,
            "optimal_action": optimal_action
        }
        
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
    regular_model_name = "deepseek-reasoner"
    meta_model_name = "deepseek-reasoner"
    
    print(f"Loading data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    base_results_dir = "synthetic_reasoning_steps/results"
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)
    
    if meta_model_name == "deepseek-reasoner":
        safe_model_name = "DeepSeek-R1"  
    elif "/" in meta_model_name:
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
                            "sample_of_100_postflop.json")
    # Adjust batch size based on your API rate limits and system capabilities
    main(data_path, batch_size=50)
