# no_batching_RS_generation_pipeline.py

'''
Usage:
    python synthetic_reasoning_steps/no_batching_RS_generation_pipeline.py
'''

import json
from openai import OpenAI
import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_reasoning_prompts import (
    prompt_proposer,
    prompt_board_verifier,
    prompt_range_estimation_verifier,
    prompt_meta_verifier,
)

# Initialize DeepSeek client
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
    # Create messages in the format expected by DeepSeek
    messages = [{"role": role, "content": prompt_text}]
    
    # Make the API call with a safe max_tokens value
    # We estimate conservatively to avoid token limit errors
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4000  # Conservative default that should work for most prompts
    )
    
    # For models with reasoning capability, we can access both
    try:
        # Try to get reasoning content (chain-of-thought)
        reasoning_content = completion.choices[0].message.reasoning_content
        content = completion.choices[0].message.content
        # Return both reasoning content and final content
        return reasoning_content, content
    except AttributeError:
        # If the model doesn't support reasoning_content, just return the regular content
        return None, completion.choices[0].message.content

def main(data_path):
    regular_model_name = "deepseek-reasoner"
    meta_model_name = "deepseek-reasoner"
    
    print(f"Loading data from: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Create a "results" directory if it doesn't exist.
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
    
    # Check if results file already exists to support resuming
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
    
    # Process each example in the data
    for i in range(starting_index, len(data)):
        example = data[i]
        print(f"\n--- Processing datapoint {i+1}/{len(data)} ---")
        
        # Extract the game state and optimal action from the example
        gamestate = example.get("instruction", "")
        optimal_action = example.get("output", "")
        
        # Dictionary to store all outputs for this example
        example_results = {}
        
        # -----------------------------
        # Step 1: Proposer Module
        # -----------------------------
        print(f"  Step 1/4: Running Proposer Module...")
        proposer_prompt = prompt_proposer(gamestate, optimal_action)
        proposer_reasoning, proposer_content = call_model(proposer_prompt, model=regular_model_name)
        print(f"  ✓ Proposer complete")
        
        # Store the proposer output
        example_results["proposer_output"] = {
            "reasoning_content": proposer_reasoning,
            "final_response": proposer_content,
            "gamestate": gamestate,
            "optimal_action": optimal_action
        }
        # We'll use the final response as the proposed reasoning.
        proposed_reasoning = proposer_content
        
        # -----------------------------
        # Step 2: Board Analysis Verifier
        # -----------------------------
        print(f"  Step 2/4: Running Board Analysis Verifier...")
        board_prompt = prompt_board_verifier(gamestate, optimal_action, proposed_reasoning)
        board_reasoning, board_content = call_model(board_prompt, model=regular_model_name)
        print(f"  ✓ Board Analysis complete")
        
        example_results["board_output"] = {
            "reasoning_content": board_reasoning,
            "final_response": board_content,
            "gamestate": gamestate,
            "optimal_action": optimal_action
        }
        
        # -----------------------------
        # Step 3: Opponent Range Estimation Verifier
        # -----------------------------
        print(f"  Step 3/4: Running Range Estimation Verifier...")
        range_estimation_prompt = prompt_range_estimation_verifier(gamestate, optimal_action, proposed_reasoning)
        range_reasoning, range_content = call_model(range_estimation_prompt, model=regular_model_name)
        print(f"  ✓ Range Estimation complete")
        
        example_results["range_estimation_output"] = {
            "reasoning_content": range_reasoning,
            "final_response": range_content,
            "gamestate": gamestate,
            "optimal_action": optimal_action
        }
        
        # -----------------------------
        # Step 4: Meta Verifier Module
        # -----------------------------
        print(f"  Step 4/4: Running Meta Verifier Module...")
        meta_prompt = prompt_meta_verifier(
            gamestate,
            optimal_action,
            proposed_reasoning,
            board_content,
            range_content
        )
        meta_reasoning, meta_content = call_model(meta_prompt, model=meta_model_name)
        print(f"  ✓ Meta Verification complete")
        
        example_results["meta_output"] = {
            "reasoning_content": meta_reasoning,
            "final_response": meta_content,
            "gamestate": gamestate,
            "optimal_action": optimal_action
        }
        
        # Add this example's results to the overall results list
        all_results.append(example_results)
        print(f"✓ Example {i+1}/{len(data)} processing complete")
        
        # Save intermediate results after each example
        with open(results_file_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"  Results saved to file ({len(all_results)}/{len(data)} examples processed)")
    
    print("\n=================================================")
    print(f"Pipeline processing complete! All {len(all_results)}/{len(data)} examples processed.")
    print(f"Results saved to: {results_file_path}")
    print("=================================================")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "pokerbench_data", 
                            "sample_of_2_postflop.json")
    main(data_path)
