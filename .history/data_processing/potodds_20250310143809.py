# potodds.py

# processes all takes in csv, for each gamestate checks if potodds is applicable (aka does "call" exist in available_moves column?). if yes, it calculates the pot odds for that gamestate.
# how to run the script: python potodds.py /path/to/csv_directory --call_amount 10 --output_dir /path/to/output_directory

import pandas as pd
import ast
import os
import glob
import argparse
import re

def is_pot_odds_applicable(row):
    """
    Returns True if pot odds are applicable, i.e. if available_moves contains a 'call' option.
    """
    try:
        moves = ast.literal_eval(row['available_moves'])
    except Exception:
        moves = []
    return any(move.strip().lower().startswith("call") for move in moves)

def get_call_amount_postflop(postflop_action, hero_pos):
    """
    Given the postflop_action string and hero's position (IP or OOP), 
    extract the call amount from the opponent's last bet or raise.
    If hero is IP, opponent actions start with "OOP_".
    If hero is OOP, opponent actions start with "IP_".
    Searches for tokens starting with opponent's prefix followed by "BET" or "RAISE" and extracts the numeric value.
    Returns a float call amount, or None if not found.
    """
    opponent_prefix = "OOP_" if hero_pos.upper() == "IP" else "IP_"
    # Split the action string by "/" into tokens.
    tokens = postflop_action.split("/")
    bet_tokens = [token for token in tokens if token.startswith(opponent_prefix + "BET") or token.startswith(opponent_prefix + "RAISE")]
    if not bet_tokens:
        return None
    # Take the last occurrence and extract numeric part.
    last_token = bet_tokens[-1]
    # Use regex to extract number (can be integer or float)
    match = re.search(r'(\d+(\.\d+)?)', last_token)
    if match:
        return float(match.group(1))
    return None

def get_call_amount_preflop(prev_line, available_moves):
    """
    For a preflop dataset, attempt to extract the call amount from the prev_line string.
    If prev_line is empty, try to extract from available_moves.
    The heuristic here is to split the string by "/" and take the last token containing 'bb'
    (e.g., "13.0bb") and parse the numeric part.
    Returns a float call amount or None if not found.
    """
    source = prev_line if isinstance(prev_line, str) and prev_line.strip() else None
    if source is None:
        # Fallback to available_moves (if they include a bet size like '3.0bb')
        try:
            moves = ast.literal_eval(available_moves)
            # Filter tokens that contain 'bb'
            bb_moves = [move for move in moves if "bb" in move.lower()]
            if bb_moves:
                source = bb_moves[-1]
        except Exception:
            return None

    if source is None:
        return None

    # Split by "/" if applicable, or if it's just a token then use it directly.
    tokens = source.split("/") if "/" in source else [source]
    # Filter tokens containing "bb" (case-insensitive)
    bb_tokens = [token for token in tokens if "bb" in token.lower()]
    if not bb_tokens:
        return None
    # Take the last token and extract numeric part
    last_token = bb_tokens[-1]
    match = re.search(r'(\d+(\.\d+)?)', last_token)
    if match:
        return float(match.group(1))
    return None

def process_csv_file(file_path):
    """
    Process a single CSV file.
    - Determines if the file is preflop or postflop based on filename.
    - For rows where available_moves include 'call', extracts the call amount from the appropriate column.
    - Computes pot odds as: call_amount / (pot_size + call_amount)
    - Adds columns for pot_odds_applicable, call_amount, and pot_odds.
    - Saves the processed DataFrame to a new CSV with '_labeled' appended to the filename.
    """
    df = pd.read_csv(file_path)
    
    # Determine file type based on filename (case-insensitive)
    filename = os.path.basename(file_path).lower()
    is_preflop = "preflop" in filename
    is_postflop = "postflop" in filename
    
    # Label gamestates where pot odds are applicable
    df['pot_odds_applicable'] = df.apply(is_pot_odds_applicable, axis=1)
    
    # Function to extract call amount dynamically
    def extract_call_amount(row):
        if not row['pot_odds_applicable']:
            return None
        if is_postflop:
            # Use postflop_action column and hero_position
            postflop_action = row.get('postflop_action', "")
            hero_pos = row.get('hero_position', "").strip()
            return get_call_amount_postflop(postflop_action, hero_pos)
        elif is_preflop:
            # Use prev_line column (fallback to available_moves if needed)
            prev_line = row.get('prev_line', "")
            available_moves = row.get('available_moves', "")
            return get_call_amount_preflop(prev_line, available_moves)
        else:
            # If file type not determined, return None
            return None
    
    # Apply extraction function to each row
    df['call_amount'] = df.apply(extract_call_amount, axis=1)
    
    # Compute pot odds where applicable: pot odds = call_amount / (pot_size + call_amount)
    def compute_row_pot_odds(row):
        call_amt = row['call_amount']
        pot_size = row['pot_size']
        if call_amt is None or call_amt <= 0:
            return None
        return call_amt / (pot_size + call_amt)
    
    df.loc[df['pot_odds_applicable'], 'pot_odds'] = df.loc[df['pot_odds_applicable']].apply(compute_row_pot_odds, axis=1)
    
    # Save processed CSV
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    output_file = os.path.join(os.path.dirname(file_path), f"{name}_labeled{ext}")
    df.to_csv(output_file, index=False)
    print(f"Processed {file_path} -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process CSV files to label pot odds eligibility and compute call amounts dynamically for preflop/postflop data.")
    parser.add_argument("input_dir", type=str, help="Directory containing CSV files")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in directory: {input_dir}")
        return
    
    for csv_file in csv_files:
        process_csv_file(csv_file)

if __name__ == "__main__":
    main()