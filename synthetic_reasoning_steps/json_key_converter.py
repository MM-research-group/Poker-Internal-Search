#!/usr/bin/env python3
import json
import os
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_json_keys(input_file, output_file=None, old_key="instruction", new_key="input"):
    """
    Convert JSON files by renaming keys.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str, optional): Path to output JSON file. If None, will modify the input file.
        old_key (str): The key name to replace
        new_key (str): The new key name
    
    Returns:
        int: Number of items processed
    """
    # Default output to input if not specified
    if output_file is None:
        output_file = input_file
    
    # Read the input JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading input file {input_file}: {e}")
        return 0
    
    item_count = 0
    modified_count = 0
    
    # Handle both list and dictionary formats
    if isinstance(data, list):
        # Process list of objects
        for item in data:
            item_count += 1
            if old_key in item:
                item[new_key] = item.pop(old_key)
                modified_count += 1
    elif isinstance(data, dict):
        # Process single object
        item_count = 1
        if old_key in data:
            data[new_key] = data.pop(old_key)
            modified_count = 1
        
        # Also check for nested objects
        for key, value in data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and old_key in item:
                        item[new_key] = item.pop(old_key)
                        modified_count += 1
            elif isinstance(value, dict) and old_key in value:
                value[new_key] = value.pop(old_key)
                modified_count += 1
    
    # Write the modified data to the output file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully processed {item_count} items, modified {modified_count} items")
        logger.info(f"Output saved to {output_file}")
        return item_count
    except Exception as e:
        logger.error(f"Error writing to output file {output_file}: {e}")
        return 0

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert JSON file by renaming keys.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output JSON file (default: overwrite input)')
    parser.add_argument('--old-key', type=str, default="instruction",
                        help='Original key name to replace (default: "instruction")')
    parser.add_argument('--new-key', type=str, default="input",
                        help='New key name (default: "input")')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Process the file
    convert_json_keys(
        input_file=args.input,
        output_file=args.output,
        old_key=args.old_key,
        new_key=args.new_key
    )

if __name__ == "__main__":
    main() 

'''

Example Usage:
    python synthetic_reasoning_steps/json_key_converter.py --input /home/xuandong/mnt/poker/Poker-Internal-Search/pokerbench_data/withpotodds_postflop_10k_test_set.json --output /home/xuandong/mnt/poker/Poker-Internal-Search/pokerbench_data/withpotodds_postflop_10k_test_set_renamed.json

'''