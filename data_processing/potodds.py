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

    # Get the last opponent's bet/raise
    last_token = bet_tokens[-1]
    match = re.search(r'(\d+(\.\d+)?)', last_token)

    if match:
        opponent_bet = float(match.group(1))
        return max(0, opponent_bet - hero_contribution)  # Ensure non-negative value

    return None

def get_call_amount_preflop(prev_line, available_moves, hero_pos, initial_stack=100.0):
    """
    Extract the call amount in preflop situations based on prior actions.

    Logic:
      - If prev_line is empty or invalid:
            - Assume the call amount is the Big Blind (1.0bb) minus the hero's contribution.
      - Else:
            - Split prev_line by "/" and search for tokens containing "bb" or "allin" (case-insensitive).
            - Use the last such token.
            - If the token contains "allin", compute call amount as:
                  initial_stack - hero's contribution.
            - If it contains "bb", extract the numeric value and subtract hero's contribution.
      - for definition of hero's contribution as per 'get_hero_contribution()'.

    Returns:
        Float: The required call amount (non-negative).
    """
    if not prev_line or not isinstance(prev_line, str):
        return 1.0 - get_hero_contribution(prev_line, hero_pos)

    tokens = prev_line.split("/")
    # Gather tokens containing either "bb" or "allin"
    valid_tokens = [(i, token) for i, token in enumerate(tokens)
                    if "bb" in token.lower() or "allin" in token.lower()]
    
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
            final_raise = float(match.group(1))
            # Subtract the hero's prior contribution
            hero_contrib = get_hero_contribution(prev_line, hero_pos)
            call_amount = final_raise - hero_contrib
            # In case hero_already_contrib >= final_raise, ensure it doesn't go negative
            return max(call_amount, 0.0)
    return None

def get_hero_contribution(prev_line, hero_pos):
    """
    Logic:
      - Use latest occurrence of hero.
      - If it's 'bb', take amount.
      - If it's 'call', backtrack to find last bet size(last raise or BB's 1bb).
      - If no explicit action, default SB = 0.5bb, BB = 1bb.
    """
    if not prev_line or not isinstance(prev_line, str):
        return 0.5 if hero_pos.upper() == "SB" else 1.0 if hero_pos.upper() == "BB" else 0.0

    tokens = prev_line.split("/")
    hero_contrib = None

    # Search for latest hero action
    for i in range(len(tokens)):
        if tokens[i].strip().lower() == hero_pos.lower():
            if i + 1 < len(tokens):
                next_token = tokens[i + 1].lower()
                if "bb" in next_token:
                    match = re.search(r'(\d+(\.\d+)?)', next_token)
                    if match:
                        hero_contrib = float(match.group(1))
                elif "call" in next_token:
                    hero_contrib = find_last_bet(tokens[:i])

    # No explicit action → use blind assumption
    if hero_contrib is None:
        if hero_pos.upper() == "SB":
            hero_contrib = 0.5
        elif hero_pos.upper() == "BB":
            hero_contrib = 1.0
        else:
            hero_contrib = 0.0

    return hero_contrib


def find_last_bet(tokens):
    """
    Backtrack tokens to find last bet/raise amount.
    """
    for j in reversed(range(len(tokens))):
        token = tokens[j].lower()
        if "bb" in token:
            match = re.search(r'(\d+(\.\d+)?)', token)
            if match:
                return float(match.group(1))
    return 1.0

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
            # Use prev_line column and hero_pos (rather than hero_position)
            prev_line = row.get('prev_line', "")
            available_moves = row.get('available_moves', "")
            hero_pos = row.get('hero_pos', "").strip()  # <-- Extract hero_pos for preflop
            return get_call_amount_preflop(prev_line, available_moves, hero_pos)

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

     # Compute pot odds ratio: (pot_size + call_amount) / call_amount
    def compute_pot_odds_ratio(row):
        call_amt = row['call_amount']
        pot_size = row['pot_size']
        if call_amt is None or call_amt <= 0:
            return None
        return f"{(pot_size + call_amt) / call_amt} : 1"

    df.loc[df['pot_odds_applicable'], 'pot_odds_ratio'] = df.loc[df['pot_odds_applicable']].apply(compute_pot_odds_ratio, axis=1)
    
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