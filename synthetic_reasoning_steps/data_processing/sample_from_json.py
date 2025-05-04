import json
import random
import argparse
import os

'''
Usage:
    python synthetic_reasoning_steps/data_processing/sample_from_json.py /home/xuandongz/mnt/MMteam_cs194/Poker-Internal-Search/synthetic_reasoning_steps/pokerbench_data/withpotodds_postflop_500k_train_set.json --output_file /home/xuandongz/mnt/MMteam_cs194/Poker-Internal-Search/synthetic_reasoning_steps/pokerbench_data/withpotodds_postflop_60k_sampled.json --seed 42

    python synthetic_reasoning_steps/data_processing/sample_from_json.py /home/xuandongz/mnt/MMteam_cs194/Poker-Internal-Search/synthetic_reasoning_steps/pokerbench_data/withpotodds_postflop_500k_train_set.json --output_file /home/xuandongz/mnt/MMteam_cs194/Poker-Internal-Search/synthetic_reasoning_steps/pokerbench_data/sample_of_100_postflop.json --seed 42 --sample_size 100    
'''

def sample_json_data(input_file, output_file, sample_size=60000):
    """
    Randomly sample data points from a JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file
        sample_size (int): Number of samples to take
    """
    # Load the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Check if we have enough data for sampling
    total_data_points = len(data)
    if total_data_points <= sample_size:
        print(f"Warning: Input file only contains {total_data_points} data points, which is less than or equal to the requested sample size of {sample_size}.")
        sampled_data = data
    else:
        # Randomly sample from the data
        sampled_data = random.sample(data, sample_size)
        print(f"Sampled {sample_size} out of {total_data_points} data points.")
    
    # Write the sampled data to the output file
    with open(output_file, 'w') as f:
        json.dump(sampled_data, f, indent=2)
    
    print(f"Sampled data saved to: {output_file}")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Randomly sample data points from a JSON file.')
    parser.add_argument('input_file', help='Path to the input JSON file')
    parser.add_argument('--output_file', help='Path to the output JSON file (default: input_filename_sampled.json)')
    parser.add_argument('--sample_size', type=int, default=60000, help='Number of samples to take (default: 60000)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # If output file not specified, generate a default name
    if args.output_file is None:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_sampled{ext}"
    
    # Sample the data
    sample_json_data(args.input_file, args.output_file, args.sample_size)
