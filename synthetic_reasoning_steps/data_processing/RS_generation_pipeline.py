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

def main():
    # Define the models to use.
    regular_model_name = "deepseek-reasoner"
    meta_model_name = "deepseek-reasoner"
    
    # Load data from local JSON file
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "pokerbench_data", 
                            "sample_of_3_postflop.json")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Get the first example from the data
    example = data[0]
    
    # Extract the game state and optimal action from the example
    gamestate = example.get("instruction", "")
    optimal_action = example.get("output", "")
    
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
    
    # -----------------------------
    # Step 1: Proposer Module
    # -----------------------------
    proposer_prompt = prompt_proposer(gamestate, optimal_action)
    proposer_reasoning, proposer_content = call_model(proposer_prompt, model=regular_model_name)
    
    # Instead of parsing JSON, we simply save the raw string response.
    proposer_output = {
        "reasoning_content": proposer_reasoning,
        "final_response": proposer_content,
        "gamestate": gamestate,
        "optimal_action": optimal_action
    }
    # We'll use the final response as the proposed reasoning.
    proposed_reasoning = proposer_content
    
    # Save the proposer output.
    with open(os.path.join(model_results_dir, "proposer_output.json"), "w") as f:
        json.dump(proposer_output, f, indent=4)
    
    # -----------------------------
    # Step 2: Board Analysis Verifier
    # -----------------------------
    board_prompt = prompt_board_verifier(gamestate, optimal_action, proposed_reasoning)
    board_reasoning, board_content = call_model(board_prompt, model=regular_model_name)
    print(board_reasoning)
    
    board_output = {
        "reasoning_content": board_reasoning,
        "final_response": board_content,
        "gamestate": gamestate,
        "optimal_action": optimal_action
    }
    
    # Save the board analysis verifier output.
    with open(os.path.join(model_results_dir, "board_output.json"), "w") as f:
        json.dump(board_output, f, indent=4)
    
    # -----------------------------
    # Step 3: Opponent Range Estimation Verifier
    # -----------------------------
    range_estimation_prompt = prompt_range_estimation_verifier(gamestate, optimal_action, proposed_reasoning)
    range_reasoning, range_content = call_model(range_estimation_prompt, model=regular_model_name)
    
    range_estimation_output = {
        "reasoning_content": range_reasoning,
        "final_response": range_content,
        "gamestate": gamestate,
        "optimal_action": optimal_action
    }
    
    # Save the opponent range estimation verifier output.
    with open(os.path.join(model_results_dir, "range_estimation_output.json"), "w") as f:
        json.dump(range_estimation_output, f, indent=4)
    
    # -----------------------------
    # Step 4: Meta Verifier Module
    # -----------------------------
    meta_prompt = prompt_meta_verifier(
        gamestate,
        optimal_action,
        proposed_reasoning,
        board_content,
        range_content
    )
    meta_reasoning, meta_content = call_model(meta_prompt, model=meta_model_name)
    
    meta_output = {
        "reasoning_content": meta_reasoning,
        "final_response": meta_content,
        "gamestate": gamestate,
        "optimal_action": optimal_action
    }
    
    # Save the meta verifier output.
    with open(os.path.join(model_results_dir, "meta_output.json"), "w") as f:
        json.dump(meta_output, f, indent=4)
    
    print("Pipeline processing complete. Check the 'results/{}' directory for each module's output.".format(safe_model_name))

if __name__ == "__main__":
    main()
