#!/usr/bin/env python3

"""
Convert existing all_results.json file to individual checkpoint files
compatible with the new checkpointing system.

Usage:
    python synthetic_reasoning_steps/data_processing/convert_to_checkpoints.py \
        --input synthetic_reasoning_steps/results/gemini-2.5-flash-preview-04-17/all_results.json \
        --batch-size 100
"""

import os
import sys
import json
import argparse
import datetime
import shutil
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert existing results file to checkpoint format')
    parser.add_argument('--input', '-i', required=True, help='Path to the existing all_results.json file')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Batch size to use for checkpoints (default: 100)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing checkpoint files')
    return parser.parse_args()

def backup_file(file_path):
    """Create a backup of a file with timestamp"""
    if os.path.exists(file_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.{timestamp}.bak"
        
        try:
            shutil.copy2(file_path, backup_path)
            print(f"✓ Created backup of file: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"⚠️ Warning: Failed to create backup: {e}")
    
    return None

def main():
    args = parse_arguments()
    input_file = args.input
    batch_size = args.batch_size
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)
    
    # Create backup of input file
    backup_file(input_file)
    
    # Load existing results
    print(f"Loading existing results from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            all_results = json.load(f)
            
        print(f"Loaded {len(all_results)} examples from existing results file")
    except Exception as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Determine model directory and create checkpoints directory
    input_path = Path(input_file)
    model_dir = input_path.parent
    checkpoints_dir = model_dir / "checkpoints"
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Checkpoints will be stored in: {checkpoints_dir}")
    
    # Convert results to checkpoint files
    total_examples = len(all_results)
    batch_count = 0
    examples_processed = 0
    
    for batch_start in range(0, total_examples, batch_size):
        batch_end = min(batch_start + batch_size, total_examples)
        batch_results = all_results[batch_start:batch_end]
        
        checkpoint_file = os.path.join(checkpoints_dir, f"checkpoint_{batch_start}-{batch_end}.json")
        
        # Skip if file exists and overwrite not specified
        if os.path.exists(checkpoint_file) and not args.overwrite:
            print(f"Skipping existing checkpoint file: {checkpoint_file} (use --overwrite to force)")
            examples_processed += len(batch_results)
            batch_count += 1
            continue
        
        # Save batch to checkpoint file
        with open(checkpoint_file, 'w') as f:
            json.dump(batch_results, f, indent=4)
        
        examples_processed += len(batch_results)
        batch_count += 1
        
        print(f"Created checkpoint {batch_count}: {checkpoint_file} ({batch_end}/{total_examples} examples)")
    
    print(f"\n✓ Conversion complete! Created {batch_count} checkpoint files with {examples_processed} examples")
    print(f"✓ You can now run the main pipeline and it will use these checkpoints to resume from where you left off")

if __name__ == "__main__":
    main() 