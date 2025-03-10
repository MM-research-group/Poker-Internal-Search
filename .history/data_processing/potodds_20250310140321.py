# potodds.py

# takes in csv, for each gamestate checks if potodds is applicable (aka does "call" exist in ). if yes, it calculates the pot odds for that gamestate.

import pandas as pd
import ast

def is_pot_odds_applicable(row):
    """
    Returns True if the gamestate is facing a bet,
    meaning that 'Call' (case-insensitive) appears in available_moves.
    """
    try:
        moves = ast.literal_eval(row['available_moves'])
    except Exception as e:
        moves = []
    return any(move.strip().lower().startswith("call") for move in moves)

def compute_pot_odds(pot_size, call_amount):
    """
    Computes pot odds as the ratio of the call amount to the total pot after calling.
    Formula: call_amount / (pot_size + call_amount)
    Returns None if call_amount is invalid.
    """
    if call_amount is None or call_amount <= 0:
        return None
    return call_amount / (pot_size + call_amount)