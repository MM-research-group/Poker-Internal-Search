# test_pipeline.py

import json
from together import Together
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_reasoning_prompts import (
    prompt_proposer,
    prompt_board_verifier,
    prompt_range_estimation_verifier,
    prompt_meta_verifier,
)

client = Together()

def call_model(prompt_text, role="user", model="deepseek-ai/DeepSeek-R1"):
    """
    Calls the Together API with the given prompt and returns the model response as a string.
    This function expects the prompt to be provided as a string.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": role, "content": prompt_text}],
    )
    return completion.choices[0].message.content

def main():
    # Define the model name to use.
    model_name = "deepseek-ai/DeepSeek-R1"
    # Create a filesystem-friendly version of the model name (replace '/' with '_').
    safe_model_name = model_name.replace("/", "_")
    
    # Load data from local JSON file
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "pokerbench_data", 
                            "withpotodds_postflop_500k_train_set.json")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Get the first example from the data
    example = data[0]
    
    # Extract the game state and optimal action from the example
    gamestate = example.get("instruction", "")
    optimal_action = example.get("output", "")
    
    # Create a "results" directory if it doesn't exist.
    base_results_dir = "results"
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)
    
    # Create a subdirectory for the current model.
    model_results_dir = os.path.join(base_results_dir, safe_model_name)
    if not os.path.exists(model_results_dir):
        os.makedirs(model_results_dir)
    
    # -----------------------------
    # Step 1: Proposer Module
    # -----------------------------
    proposer_prompt = prompt_proposer(gamestate, optimal_action)
    proposer_response = call_model(proposer_prompt, model=model_name)
    
    # Instead of parsing JSON, we simply save the raw string response.
    proposer_output = {
        "raw_response": proposer_response,
        "gamestate": gamestate,
        "optimal_action": optimal_action
    }
    # We'll use the raw response as the proposed reasoning.
    proposed_reasoning = proposer_response
    
    # Save the proposer output.
    with open(os.path.join(model_results_dir, "proposer_output.json"), "w") as f:
        json.dump(proposer_output, f, indent=4)
    
    # -----------------------------
    # Step 2: Board Analysis Verifier
    # -----------------------------
    board_prompt = prompt_board_verifier(gamestate, optimal_action, proposed_reasoning)
    board_response = call_model(board_prompt, model=model_name)
    
    board_output = {
        "raw_response": board_response,
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
    range_estimation_response = call_model(range_estimation_prompt, model=model_name)
    
    range_estimation_output = {
        "raw_response": range_estimation_response,
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
        board_response,      # pass the raw board response
        range_estimation_response  # pass the raw opponent range response
    )
    meta_response = call_model(meta_prompt, model=model_name)
    
    meta_output = {
        "raw_response": meta_response,
        "gamestate": gamestate,
        "optimal_action": optimal_action
    }
    
    # Save the meta verifier output.
    with open(os.path.join(model_results_dir, "meta_output.json"), "w") as f:
        json.dump(meta_output, f, indent=4)
    
    print("Pipeline processing complete. Check the 'results/{}' directory for each module's output.".format(safe_model_name))

if __name__ == "__main__":
    main()
