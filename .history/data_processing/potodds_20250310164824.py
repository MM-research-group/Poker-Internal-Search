# potodds.py

# this file processes all csvs in specified directory. 
# for both preflop and postflop gamestate configurations
# it checks if potodds are applicable (aka does "call" exist in available_moves column?). 
# if yes, it calculates the pot odds for that gamestate and saves it.

# how to run the script: python data_processing/potodds.py /path/to/csv_directory

import pandas as pd
import ast
import os
import glob
import argparse
import re

# verified by michael
def is_pot_odds_applicable(row):
    """
    Returns True if pot odds are applicable, i.e. if available_moves contains a 'call' option.
    """
    try:
        moves = ast.literal_eval(row['available_moves'])
    except Exception:
        moves = []
    return any(move.strip().lower().startswith("call") for move in moves)

# verified by michael
def get_call_amount_postflop(postflop_action, hero_pos):
    """
    Extract the call amount from the opponent's last bet or raise in the most recent betting round.
    If "dealcards" exists, only consider actions after the last "dealcards/{card}" token.
    """
    opponent_prefix = "OOP_" if hero_pos.upper() == "IP" else "IP_"
    hero_prefix = "IP_" if hero_pos.upper() == "IP" else "OOP_"

    tokens = postflop_action.split("/")
    dealcard_indices = [i for i, token in enumerate(tokens) if token.startswith("dealcards")]

    if dealcard_indices:
        last_deal_idx = max(dealcard_indices)
        relevant_actions = tokens[last_deal_idx + 2:]
    else:
        relevant_actions = tokens  # If no "dealcards", consider entire action string

    hero_contribution = 0.0
    bet_tokens = []

    for token in relevant_actions:
        if token.startswith(opponent_prefix + "BET") or token.startswith(opponent_prefix + "RAISE"):
            bet_tokens.append(token)
        elif token.startswith(hero_prefix + "BET") or token.startswith(hero_prefix + "RAISE"):
            match = re.search(r'(\d+(\.\d+)?)', token)
            if match:
                hero_contribution += float(match.group(1))

    if not bet_tokens:
        return None

    last_token = bet_tokens[-1]
    match = re.search(r'(\d+(\.\d+)?)', last_token)

    if match:
        opponent_bet = float(match.group(1))
        return max(0, opponent_bet - hero_contribution)

    return None

def get_call_amount_preflop(prev_line, hero_pos, initial_stack=100.0):
    """
    For a preflop dataset, extract the call amount from the prev_line string.
    
    Heuristic:
      - Split the string by "/" and search for tokens that contain either "bb" or "allin" (case-insensitive).
      - Use the token that appears last in the sequence.
      - If that token contains "allin", compute the call amount as the hero's remaining stack:
            remaining_stack = initial_stack - (sum of all hero contributions from prev_line)
      - If the token contains "bb", extract and return the numeric value from that token.
      
    Returns a float representing the call amount or None if not found.
    """
    if not isinstance(prev_line, str) or not prev_line.strip():
        return None

    tokens = prev_line.split("/")
    # Gather tokens containing either "bb" or "allin"
    valid_tokens = [(i, token) for i, token in enumerate(tokens)
                    if "bb" in token.lower() or "allin" in token.lower()]
    if not valid_tokens:
        return None
    
    # Use the token with the highest index (i.e. the last occurrence)
    _, last_token = max(valid_tokens, key=lambda x: x[0])
    token_lower = last_token.lower()
    
    if "allin" in token_lower:
        # Sum all hero contributions (multiple actions may have been made)
        hero_contrib = get_hero_contribution(prev_line, hero_pos)
        remaining_stack = initial_stack - hero_contrib
        return remaining_stack
    elif "bb" in token_lower:
        match = re.search(r'(\d+(\.\d+)?)', last_token)
        if match:
            return float(match.group(1))
    return None

def get_hero_contribution(prev_line, hero_pos):
    """
    Parse the prev_line string to find all occurrences of hero_pos (case-insensitive)
    and sum up the numeric values in the token immediately following each occurrence that contains 'bb'.
    This represents the total amount the hero has already contributed.
    """
    if not prev_line or not isinstance(prev_line, str):
        return 0.0
    tokens = prev_line.split("/")
    hero_contrib = 0.0
    for i, token in enumerate(tokens):
        if token.strip().lower() == hero_pos.lower():
            # Check the next token for a bet amount if it exists
            if i + 1 < len(tokens):
                next_token = tokens[i+1]
                if "bb" in next_token.lower():
                    match = re.search(r'(\d+(\.\d+)?)', next_token)
                    if match:
                        hero_contrib += float(match.group(1))
    return hero_contrib

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
    
    filename = os.path.basename(file_path).lower()
    is_preflop = "preflop" in filename
    is_postflop = "postflop" in filename
    
    df['pot_odds_applicable'] = df.apply(is_pot_odds_applicable, axis=1)
    
    def extract_call_amount(row):
        if not row['pot_odds_applicable']:
            return None

        if is_postflop: # verified by michael
            postflop_action = row.get('postflop_action', "")
            hero_pos = row.get('hero_position', "").strip()
            return get_call_amount_postflop(postflop_action, hero_pos)

        elif is_preflop:
            # Use prev_line column and hero_pos (rather than hero_position)
            prev_line = row.get('prev_line', "")
            hero_pos = row.get('hero_pos', "").strip()  # <-- Extract hero_pos for preflop
            return get_call_amount_preflop(prev_line,, hero_pos)

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
    output_file = os.path.join(os.path.dirname(file_path), f"{name}_withpotodds{ext}")
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