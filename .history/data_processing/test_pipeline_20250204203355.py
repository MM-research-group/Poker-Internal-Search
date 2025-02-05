# test_pipeline.py

import json
from together import Together
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_reasoning_prompts import (
    prompt_proposer,
    prompt_math_board_verifier,
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
    # Load the PokerBench dataset and select a single datapoint.
    ds = load_dataset("RZ412/PokerBench")
    example = ds["train"][0]  # Use the first datapoint for testing

    # Extract the game state and optimal action from the example.
    gamestate = example.get("instruction", "")
    optimal_action = example.get("output", "")
    
    # Create a "results" directory if it doesn't exist.
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # -----------------------------
    # Step 1: Proposer Module
    # -----------------------------
    proposer_prompt = prompt_proposer(gamestate, optimal_action)
    proposer_response = call_model(proposer_prompt)
    
    # Instead of parsing JSON, we simply save the raw string response.
    proposer_output = {
        "raw_response": proposer_response,
        "gamestate": gamestate,
        "optimal_action": optimal_action
    }
    # We'll use the raw response as the proposed reasoning.
    proposed_reasoning = proposer_response
    
    # Save the proposer output.
    with open(os.path.join(results_dir, "proposer_output.json"), "w") as f:
        json.dump(proposer_output, f, indent=4)
    
    # -----------------------------
    # Step 2: Math & Board Analysis Verifier
    # -----------------------------
    math_board_prompt = prompt_math_board_verifier(gamestate, optimal_action, proposed_reasoning)
    math_board_response = call_model(math_board_prompt)
    
    math_board_output = {
        "raw_response": math_board_response,
        "gamestate": gamestate,
        "optimal_action": optimal_action
    }
    
    # Save the math and board analysis verifier output.
    with open(os.path.join(results_dir, "math_board_output.json"), "w") as f:
        json.dump(math_board_output, f, indent=4)
    
    # -----------------------------
    # Step 3: Opponent Range Estimation Verifier
    # -----------------------------
    range_estimation_prompt = prompt_range_estimation_verifier(gamestate, optimal_action, proposed_reasoning)
    range_estimation_response = call_model(range_estimation_prompt)
    
    range_estimation_output = {
        "raw_response": range_estimation_response,
        "gamestate": gamestate,
        "optimal_action": optimal_action
    }
    
    # Save the opponent range estimation verifier output.
    with open(os.path.join(results_dir, "range_estimation_output.json"), "w") as f:
        json.dump(range_estimation_output, f, indent=4)
    
    # -----------------------------
    # Step 4: Meta Verifier Module
    # -----------------------------
    meta_prompt = prompt_meta_verifier(
        gamestate,
        optimal_action,
        proposed_reasoning,
        math_board_response,      # pass the raw math & board response
        range_estimation_response  # pass the raw opponent range response
    )
    meta_response = call_model(meta_prompt)
    
    meta_output = {
        "raw_response": meta_response,
        "gamestate": gamestate,
        "optimal_action": optimal_action
    }
    
    # Save the meta verifier output.
    with open(os.path.join(results_dir, "meta_output.json"), "w") as f:
        json.dump(meta_output, f, indent=4)
    
    print("Pipeline processing complete. Check the 'results' directory for each module's output.")

if __name__ == "__main__":
    main()