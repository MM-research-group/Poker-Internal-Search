# pipeline.py

import json
from together import Together
from datasets import load_dataset

from generate_reasoning_prompts import (
    prompt_proposer,
    prompt_math_board_verifier,
    prompt_range_estimation_verifier,
    prompt_meta_verifier,
)

client = Together()

def call_model(prompt_text, role="user", model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
    """
    Calls the Together API with the given prompt and returns the model response as a string.
    This function expects the prompt to be provided as a string.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": role, "content": prompt_text}],
    )
    return completion.choices[0].message.content

def run_pipeline(example):
    """
    For a given example from the dataset, run the proposer, verifiers, and meta-verifier in sequence.
    Returns the final meta-verifier JSON output or None if an error occurs.
    """
    # Extract required fields from the example (adjust field names as necessary)
    gamestate = example.get("instruction", "")
    optimal_action = example.get("optimal_action", "")
    
    # -----------------------------
    # Step 1: Proposer Module
    # -----------------------------
    proposer_prompt = prompt_proposer(gamestate, optimal_action)
    proposer_response = call_model(proposer_prompt)
    
    try:
        proposer_output = json.loads(proposer_response)
    except Exception as e:
        print("Error parsing proposer response:")
        print(proposer_response)
        return None

    # The proposer output is expected to include the key "chain_of_thought"
    proposed_reasoning = proposer_output.get("chain_of_thought", "")
    
    # -----------------------------
    # Step 2: Math & Board Analysis Verifier
    # -----------------------------
    math_board_prompt = prompt_math_board_verifier(gamestate, optimal_action, proposed_reasoning)
    math_board_response = call_model(math_board_prompt)
    
    try:
        math_board_output = json.loads(math_board_response)
    except Exception as e:
        print("Error parsing math & board verifier response:")
        print(math_board_response)
        return None

    # -----------------------------
    # Step 3: Opponent Range Estimation Verifier
    # -----------------------------
    range_estimation_prompt = prompt_range_estimation_verifier(gamestate, optimal_action, proposed_reasoning)
    range_estimation_response = call_model(range_estimation_prompt)
    
    try:
        range_estimation_output = json.loads(range_estimation_response)
    except Exception as e:
        print("Error parsing range estimation verifier response:")
        print(range_estimation_response)
        return None

    # -----------------------------
    # Step 4: Meta Verifier Module
    # -----------------------------
    meta_prompt = prompt_meta_verifier(
        gamestate,
        optimal_action,
        proposed_reasoning,
        math_board_output,
        range_estimation_output,
    )
    meta_response = call_model(meta_prompt)
    
    try:
        meta_output = json.loads(meta_response)
    except Exception as e:
        print("Error parsing meta verifier response:")
        print(meta_response)
        return None

    return meta_output

def main():
    # Load the PokerBench dataset from the Hugging Face hub.
    ds = load_dataset("RZ412/PokerBench")
    
    # For this example, assume we're processing the training split.
    results = []
    # Process a few examples from the dataset (adjust the range or add a filter as needed)
    for i, example in enumerate(ds["train"]):
        print(f"Processing example {i}...")
        final_output = run_pipeline(example)
        if final_output is not None:
            results.append(final_output)
            # Optionally, print the final chain-of-thought explanation.
            print("Final chain-of-thought explanation:")
            print(final_output.get("final_chain_of_thought", ""))
        else:
            print(f"Pipeline failed for example {i}.")
        
        # For demonstration, we process only the first 5 examples.
        if i >= 4:
            break

    # Optionally, save the results to a JSON file.
    with open("poker_reasoning_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Pipeline processing complete. Results saved to 'poker_reasoning_results.json'.")

if __name__ == "__main__":
    main()