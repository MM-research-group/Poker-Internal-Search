# generate_reasoning_prompts.py

def prompt_proposer(gamestate, optimal_action):
    prompt = """
You are a professional poker player known for your deep strategic insight. Given the following game state and knowing that the solver has determined the optimal action to be {optimal_action}, please provide a detailed chain-of-thought explanation for why this action is the best choice. In your explanation, consider the following:
- What is my opponent's range here? Is their range capped or uncapped?
- What is our hand? (value, showdown value, bluff)
- Given the opponent's range, what can we do about it?
- Risk management, pot odds, stack sizes (when necessary)
- Future street considerations and board texture

Only use factors that make sense given the current game state. Break down your reasoning step by step, ensuring that each part logically follows from the game state and standard poker strategies. Avoid speculative statements that are not grounded in poker theory.

Game State: {gamestate}
Optimal Action: {optimal_action}

Please provide your chain-of-thought explanation along with a brief summary of the main points.
    """.format(gamestate=gamestate, optimal_action=optimal_action)
    return prompt

def prompt_range_estimation_verifier(gamestate, optimal_action, proposed_reasoning):
    prompt = """
You are an expert in opponent range estimation in poker scenarios. Your task is to evaluate the portion of the chain-of-thought reasoning below that discusses the opponent's potential holdings based on the provided game state and betting actions.

Your tasks are:

Opponent Range Analysis Verification:
- Analyze the reasoning that identifies the opponent's possible holdings, considering their pre-flop, flop, and subsequent actions.
- Evaluate whether the explanation logically categorizes the opponent’s range into segments such as made hands, draws, and potential bluffs.
- Assess if the explanation provides an accurate estimation of the proportion or likelihood of various holdings (e.g., the balance between draws and made hands).
- Flag any inaccuracies, misinterpretations, or omissions in the opponent range analysis and provide corrections.

Output Requirements:
- Provide a revised version of the opponent range estimation section with corrections applied.
- Include a brief summary of the changes you made.
- If the explanation is missing an range analysis, make sure to add your analysis.

Game State: {gamestate}
Optimal Action: {optimal_action}
Proposed Reasoning: {proposed_reasoning}
    """.format(gamestate=gamestate, optimal_action=optimal_action, proposed_reasoning=proposed_reasoning)
    return prompt


def prompt_meta_verifier(gamestate, optimal_action, proposed_reasoning, board_output, range_estimation_output):
    prompt = """
You are an expert poker strategist tasked with a comprehensive review of the chain-of-thought explanation for a poker scenario. In this review, you have access to multiple sources:
1. The original proposed chain-of-thought explanation.
2. The revised board analysis output (see below).
3. The revised opponent range estimation output (see below).

Your objectives are:

Evaluate Revisions:
- Determine if the corrections make sense, if there are any.

Overall Coherence and Accuracy:
- Integrate the corrections from the board analysis and opponent range estimation outputs into a final chain-of-thought explanation.
- Ensure that every part of the final explanation is logically consistent and factually correct according to standard poker strategy.
- Verify that all board analyses, and opponent range analyses are valid and coherently merged. Make sure you're weighing all of the considerations for range analysis and determining which ones are the most important.

Hallucination and Inconsistency Detection:
- If you see any remaining hallucinations, misrepresentations, or logical inconsistencies in the combined explanation, please fix it.

Removal of Optimal Action References:
- Ensure that the final verified reasoning (chain-of-thought explanation) does not include any reference to the optimal action. Remove any such details so that the reasoning stands alone.

Final Output Requirements:
- In your final response, present only the fully revised and integrated chain-of-thought explanation that is self-contained, free of any optimal action details, and adheres to strong game theory optimal poker concepts.

Below are the inputs:

Game State: {gamestate}
Optimal Action: {optimal_action}
Original Proposed Reasoning: {proposed_reasoning}

Board Analysis Revision:
{board_output}

Opponent Range Estimation Revision:
{range_estimation_output}

Please proceed step by step in your verification.
    """.format(
        gamestate=gamestate,
        optimal_action=optimal_action,
        proposed_reasoning=proposed_reasoning,
        board_output=board_output,
        range_estimation_output=range_estimation_output
    )
    return prompt