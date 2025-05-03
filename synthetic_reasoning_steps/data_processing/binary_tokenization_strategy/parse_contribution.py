# binary_tokenization_strategy/parse_contribution.py

def parse_preflop_stack_size(preflop_action_line, starting_stack_size, position):
    known_positions = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
    players_contrib = {}
    current_bet = 0
    actions = preflop_action_line.split('/')

    # Handle pre-flop standard contributions: SB = 0.5 BB, BB = 1 BB
    players_contrib['SB'] = 0.5
    players_contrib['BB'] = 1
    players_contrib['UTG'] = 0
    players_contrib['HJ'] = 0
    players_contrib['CO'] = 0
    players_contrib['BTN'] = 0
    if preflop_action_line == "":
        return float(starting_stack_size - players_contrib[position])

    # Iterate through each action in the string
    i = 0
    while i < len(actions):
        if sum([pos in actions[i] for pos in known_positions]):
            i += 1
            continue
        if 'call' in actions[i]:
            player = actions[i-1]
            # Player calls the current bet
            amount_to_add = current_bet - players_contrib.get(player, 0)
            players_contrib[player] = players_contrib.get(player, 0) + amount_to_add
            i += 1  # Skip the 'call' keyword
        elif 'allin' in actions[i]:
            player = actions[i-1]
            # All-in is considered as 100 bb
            players_contrib[player] = starting_stack_size
            current_bet = max(current_bet, starting_stack_size)
            i += 1  # Skip the 'allin' keyword
        elif 'fold' in actions[i]:
            # On fold, do nothing as the player's current contribution remains
            i += 1  # Skip the 'fold' keyword
        elif 'check' in actions[i]:
            # On check, no changes to contributions
            i += 1  # Skip the 'check' keyword
        else:
            # This is a betting action
            player, bet_str = actions[i-1], actions[i]
            bet_amount = float(bet_str.replace('bb', ''))
            players_contrib[player] = bet_amount
            current_bet = max(current_bet, bet_amount)
            i += 1  # Move to next token which should be the action or next player

        i += 1  # General increment to move to the next token

    return float(starting_stack_size - players_contrib[position])

def parse_preflop_action(preflop_action):
    """Parses the preflop action to determine the two player positions and their initial contributions."""
    actions = preflop_action.split("/")
    positions = []
    contributions = {}

    if len(actions) == 4:  # Two-action case (e.g., "HJ/2.0bb/BB/call")
        positions = [actions[0], actions[2]]
        contributions[actions[0]] = float(actions[1].replace("bb", ""))  # First bet size
        contributions[actions[2]] = contributions[actions[0]]  # Caller matches the first bet

    elif len(actions) == 6:  # Three-action case (e.g., "SB/3.0bb/BB/10.0bb/SB/call")
        positions = [actions[0], actions[2]]
        contributions[actions[0]] = float(actions[1].replace("bb", ""))  # Initial small bet
        contributions[actions[2]] = float(actions[3].replace("bb", ""))  # Raise amount
        contributions[actions[0]] = contributions[actions[2]]  # Caller matches the raise amount

    else:
        raise ValueError("Unexpected format in preflop_action.")

    return contributions, positions

def determine_oop_ip_positions(positions):
    """Determines the OOP and IP positions based on Texas Hold'em position rules."""
    order = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]
    sorted_positions = sorted(positions, key=lambda pos: order.index(pos))
    oop_position = sorted_positions[0]  # First in the order is OOP
    ip_position = sorted_positions[1]   # Second in the order is IP
    return oop_position, ip_position

def parse_postflop_action(postflop_action, oop_position, ip_position):
    """Parses the postflop action to track contributions of OOP and IP players."""
    actions = postflop_action.split("/")
    player_position_map = {"OOP": oop_position, "IP": ip_position}
    contributions = {oop_position: 0, ip_position: 0}
    for action in actions:
        if "_" in action:
            # Example action format: IP_BET_10
            parts = action.split("_")
            player_role = parts[0]  # "IP" or "OOP"
            action_type = parts[1]  # "BET", "CALL", "RAISE", etc.
            amount = float(parts[2]) if len(parts) > 2 else 0
            
            # Map role to actual position
            player_position = player_position_map[player_role]
            
            # Update contributions based on the action type
            if action_type in ["BET", "RAISE"]:
                contributions[player_position] += amount
            elif action_type == "CALL":
                opponent_position = oop_position if player_role == "IP" else ip_position
                contributions[player_position] += contributions[opponent_position] - contributions[player_position]
            # CHECK action is ignored as it doesn't affect contribution

    return contributions

def calculate_player_contribution(preflop_action, postflop_action, hero_position):
    # Step 1: Parse the preflop action
    contributions_preflop, positions = parse_preflop_action(preflop_action)
    # print(contributions_preflop)
    # Step 2: Determine which positions are OOP and IP based on preflop positions
    oop_position, ip_position = determine_oop_ip_positions(positions)
    # print(oop_position, ip_position)
    # Step 3: Initialize contributions for each position
    contributions_preflop.setdefault(oop_position, 0)
    contributions_preflop.setdefault(ip_position, 0)
    
    # Step 4: Parse the postflop action and update contributions
    contributions_postflop = parse_postflop_action(postflop_action, oop_position, ip_position)
    # print(contributions_postflop)
    contributions = {oop_position: contributions_preflop[oop_position] + contributions_postflop[oop_position],
                     ip_position: contributions_preflop[ip_position] + contributions_postflop[ip_position]}
    
    # Step 5: Return the contribution for the specified hero position (OOP or IP)
    if hero_position == "OOP":
        return contributions[oop_position]
    elif hero_position == "IP":
        return contributions[ip_position]
    else:
        raise ValueError("Invalid hero_position. Expected 'OOP' or 'IP'.")

# Example usage
# preflop_action = "HJ/2.0bb/BB/call"
# postflop_action = "OOP_CHECK/IP_BET_10/OOP_CALL/dealcards/Jc/OOP_CHECK/IP_BET_44/OOP_RAISE_80"
# hero_position = "IP"
# print(calculate_player_contribution(preflop_action, postflop_action, hero_position))