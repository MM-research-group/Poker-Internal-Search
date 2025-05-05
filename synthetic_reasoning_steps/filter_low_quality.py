#!/usr/bin/env python3
import json
import os

# Input and output file paths
input_file = "/home/xuandong/mnt/poker/Poker-Internal-Search/synthetic_reasoning_steps/all_results_gemini-2.5-flash-preview-04-17/transformed_postflop_54785_synthetic_reasoning_steps_train_set.json"
output_file = "/home/xuandong/mnt/poker/Poker-Internal-Search/synthetic_reasoning_steps/all_results_gemini-2.5-flash-preview-04-17/final_sft_postflop_trainset.json"

# Filter criterion
filter_string = "Original Proposed Reasoning:"

def main():
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Processing file: {input_file}")
    
    # Read, filter, and write the data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("Error: Expected a JSON array at the top level.")
            return
        
        original_count = len(data)
        print(f"Original number of objects: {original_count}")
        
        # Filter out objects with the filter string in the "output" field
        filtered_data = [item for item in data if "output" in item and filter_string not in item["output"]]
        
        filtered_count = len(filtered_data)
        print(f"Filtered number of objects: {filtered_count}")
        print(f"Removed {original_count - filtered_count} objects containing '{filter_string}'")
        
        # Write the filtered data to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"Successfully wrote filtered data to {output_file}")
        
    except json.JSONDecodeError:
        print("Error: Input file contains invalid JSON.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 