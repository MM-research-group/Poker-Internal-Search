# potodds.py

# takes in csv, for each gamestate checks if potodds is applicable (aka does "call" exist in available_moves column?). if yes, it calculates the pot odds for that gamestate.

import pandas as pd
import ast
import os
import glob
import argparse

def is_pot_odds_applicable(row):
    """
    Returns True if pot odds are applicable for the gamestate.
    We assume that if any available move starts with "Call", then the player is facing a bet.
    """
    try:
        moves = ast.literal_eval(row['available_moves'])
    except Exception:
        moves = []
    # Check if any move starts with "call" (case insensitive)
    return any(move.strip().lower().startswith("call") for move in moves)

def compute_pot_odds(pot_size, call_amount):
    """
    Computes pot odds as the ratio of the call amount to the total pot size after calling.
    Formula: call_amount / (pot_size + call_amount)
    """
    if call_amount is None or call_amount <= 0:
        return None
    return call_amount / (pot_size + call_amount)

def process_csv_file(file_path, call_amount, output_dir):
    """
    Process a single CSV file:
      - Reads the CSV into a DataFrame.
      - Labels rows as eligible for pot odds.
      - Computes pot odds where applicable.
      - Saves the processed DataFrame to a new CSV in the output directory.
    """
    df = pd.read_csv(file_path)
    
    # Label gamestates where pot odds are applicable
    df['pot_odds_applicable'] = df.apply(is_pot_odds_applicable, axis=1)
    
    # Compute pot odds for rows where it is applicable
    df.loc[df['pot_odds_applicable'], 'pot_odds'] = df.loc[df['pot_odds_applicable']].apply(
        lambda row: compute_pot_odds(row['pot_size'], call_amount), axis=1)
    
    # Create an output filename by appending '_labeled' before the file extension
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    output_file = os.path.join(output_dir, f"{name}_labeled{ext}")
    
    df.to_csv(output_file, index=False)
    print(f"Processed {file_path} -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Label CSV files with pot odds eligibility and compute pot odds for applicable gamestates.")
    parser.add_argument("input_dir", type=str, help="Directory containing CSV files")
    parser.add_argument("--call_amount", type=float, default=10.0, help="Call amount for pot odds calculation (default: 10)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save processed CSV files (default: same as input_dir)")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    call_amount = args.call_amount
    output_dir = args.output_dir if args.output_dir else input_dir
    
    # Find all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in directory: {input_dir}")
        return
    
    for csv_file in csv_files:
        process_csv_file(csv_file, call_amount, output_dir)

if __name__ == "__main__":
    main()