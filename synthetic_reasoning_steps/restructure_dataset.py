#!/usr/bin/env python3
import json
import os

def restructure_json(input_file, output_file=None):
    # Default output to input if not specified
    if output_file is None:
        output_file = input_file
    
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Restructure the data
    restructured_data = []
    for item in data:
        # Get the values
        optimal_action = item.get('optimal_action', '')
        meta_response = item.get('meta_response', '')
        input_text = item.get('input', '')
        
        # Append the optimal action to the meta_response
        new_meta_response = meta_response + f"\n\nThe optimal action is: {optimal_action}"
        
        # Create the new restructured item
        restructured_item = {
            "input": input_text,
            "meta_response": new_meta_response
        }
        
        restructured_data.append(restructured_item)
    
    # Write the restructured data to the output file
    with open(output_file, 'w') as f:
        json.dump(restructured_data, f, indent=4)
    
    return len(restructured_data)

if __name__ == "__main__":
    input_file = "synthetic_reasoning_steps/sample_dataset.json"
    output_file = "synthetic_reasoning_steps/restructured_dataset.json"
    
    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        exit(1)
    
    # Create directory for output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Restructure the JSON
    items_processed = restructure_json(input_file, output_file)
    
    print(f"Successfully restructured {items_processed} items.")
    print(f"Output saved to {output_file}") 