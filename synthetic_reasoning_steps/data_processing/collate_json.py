#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

def find_json_files(directory):
    """Recursively find all JSON files in the given directory."""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def collate_json_files(json_files, output_file):
    """Collate multiple JSON files into a single JSON file."""
    result = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Add source file info
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            item['_source_file'] = file_path
                    result.extend(data)
                elif isinstance(data, dict):
                    data['_source_file'] = file_path
                    result.append(data)
        except json.JSONDecodeError:
            print(f"Error: {file_path} is not a valid JSON file. Skipping.")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return len(result)

def main():
    parser = argparse.ArgumentParser(description='Collate JSON files from a directory into a single JSON file.')
    parser.add_argument('input_dir', help='Input directory containing JSON files')
    parser.add_argument('-o', '--output', default='collated_output.json', 
                        help='Output JSON file (default: collated_output.json)')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_file = args.output
    
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        return
    
    json_files = find_json_files(input_dir)
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    items_count = collate_json_files(json_files, output_file)
    print(f"Successfully collated {items_count} items into {output_file}")

if __name__ == "__main__":
    main() 