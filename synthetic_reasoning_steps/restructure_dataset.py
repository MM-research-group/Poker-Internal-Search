#!/usr/bin/env python3
import json
import os
import sys

def transform_data(input_file):
    """
    Transform the data from the input JSON file according to specified requirements:
    1. Remove "proposer_response" and "board_range_response" keys
    2. Trim "meta_response" by removing everything before the first "\n\n"
    3. Append "optimal_action" to the end of "meta_response"
    4. Remove "optimal_action" key after appending
    5. Rename "gamestate" key to "input"
    6. Rename "meta_response" key to "output"
    
    Items with None values for required fields will be skipped.
    """
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Restructure the data
    restructured_data = []
    skipped_items = 0
    
    for item in data:
        # Get the values, skip item if required fields are None
        gamestate = item.get('gamestate')
        optimal_action = item.get('optimal_action')
        meta_response = item.get('meta_response')
        
        # Skip this item if any required field is None
        if gamestate is None or optimal_action is None or meta_response is None:
            skipped_items += 1
            continue

        # Remove keys we don't want
        if 'proposer_response' in item:
            del item['proposer_response']
        if 'board_range_response' in item:
            del item['board_range_response']
        
        # Trim meta_response - remove everything before first "\n\n"
        if '\n\n' in meta_response:
            meta_response = meta_response.split('\n\n', 1)[1]
        
        # Append the optimal action to the meta_response
        new_meta_response = meta_response + f"\n\nThe optimal action is: {optimal_action}"
        
        # Create the new restructured item
        restructured_item = {
            "input": gamestate,
            "output": new_meta_response
        }
        
        restructured_data.append(restructured_item)
    
    print(f"Skipped {skipped_items} items due to missing required fields.")
    return restructured_data

def save_transformed_data(data, input_file):
    """
    Save the transformed data to a new file in the same directory as the input file
    """
    # Generate output file path based on input file
    input_dir = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    output_filename = "transformed_" + input_filename
    output_file = os.path.join(input_dir, output_filename)
    
    # Create directory for output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the restructured data to the output file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    return output_file

def run_tests():
    """
    Create a test JSON file and test the transformation functions
    """
    test_dir = "synthetic_reasoning_steps/test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test data
    test_data = [
        {
            "gamestate": "Test scenario 1",
            "optimal_action": "Call",
            "proposer_response": "Some proposer reasoning",
            "board_range_response": "Some evaluation",
            "meta_response": "Header information\n\nActual summary to keep"
        },
        {
            "gamestate": "Test scenario 2",
            "optimal_action": "Fold",
            "proposer_response": "More proposer reasoning",
            "board_range_response": "More evaluation",
            "meta_response": "Another header\n\nSummary to keep\n\nWith multiple paragraphs"
        },
        # Add test case with None values - should be skipped
        {
            "gamestate": "Test scenario 3",
            "optimal_action": None,
            "proposer_response": "Some proposer reasoning",
            "board_range_response": "Some evaluation",
            "meta_response": "Header information\n\nActual summary to keep"
        },
        # Test case with missing meta_response - should be skipped
        {
            "gamestate": "Test scenario 4",
            "optimal_action": "Raise",
            "proposer_response": "Some proposer reasoning",
            "board_range_response": "Some evaluation",
            "meta_response": None
        }
    ]
    
    # Save test data to file
    test_file = os.path.join(test_dir, "test_input.json")
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    # Run the transformation
    transformed_data = transform_data(test_file)
    output_file = save_transformed_data(transformed_data, test_file)
    
    # Load the transformed data to verify
    with open(output_file, 'r') as f:
        result = json.load(f)
    
    # Verify requirements
    success = True
    
    # There should be only 2 items in the result (2 items with None values should be skipped)
    if len(result) != 2:
        print(f"FAIL: Expected 2 items, got {len(result)} items")
        success = False
    
    for i, item in enumerate(result):
        # Check that the right keys exist
        if not all(k in item for k in ["input", "output"]):
            print(f"FAIL: Missing required keys in item {i}")
            success = False
        
        # Check that the unwanted keys don't exist
        if any(k in item for k in ["proposer_response", "board_range_response", "optimal_action", "gamestate", "meta_response"]):
            print(f"FAIL: Unwanted keys found in item {i}")
            success = False
        
        # Verify content
        expected_input = test_data[i]["gamestate"]
        expected_output = "Actual summary to keep" if i == 0 else "Summary to keep\n\nWith multiple paragraphs"
        expected_output += f"\n\nThe optimal action is: {test_data[i]['optimal_action']}"
        
        if item["input"] != expected_input:
            print(f"FAIL: Input mismatch in item {i}")
            print(f"Expected: {expected_input}")
            print(f"Got: {item['input']}")
            success = False
            
        if item["output"] != expected_output:
            print(f"FAIL: Output mismatch in item {i}")
            print(f"Expected: {expected_output}")
            print(f"Got: {item['output']}")
            success = False
    
    if success:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED")
    
    return success

if __name__ == "__main__":
    # Check if we're in test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        # Get input file path from command line or use default
        input_file = sys.argv[1] if len(sys.argv) > 1 else "synthetic_reasoning_steps/sample_dataset.json"
        
        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file {input_file} does not exist.")
            exit(1)
        
        # Transform and save the data
        transformed_data = transform_data(input_file)
        output_file = save_transformed_data(transformed_data, input_file)
        
        print(f"Successfully transformed {len(transformed_data)} items.")
        print(f"Output saved to {output_file}") 