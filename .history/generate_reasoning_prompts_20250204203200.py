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


def prompt_math_board_verifier(gamestate, optimal_action, proposed_reasoning):
    prompt = """
You are a dedicated expert in verifying mathematical reasoning and board analysis in poker scenarios.
Below is the chain-of-thought reasoning corresponding to the provided game state. This reasoning should include both mathematical computations and board analysis. (If one or both are missing, please note that explicitly in your response.)

Your tasks are to verify the following areas within the reasoning:

Mathematical Verification:
- Identify every instance where mathematical computations or numerical reasoning is applied (e.g., probability calculations, pot odds, percentage computations).
- Verify that all arithmetic is correct and that probability or risk assessments follow standard poker math.
- Flag and explain any miscalculations or inconsistencies.

Board Analysis Verification:
- Review all sections where the board is analyzed (e.g., card counts, suit distributions, potential flushes, straight draws, or other spatial/card pattern considerations).
- Ensure that the board description (e.g., number of spades, hearts, etc.) is consistent with the actual game state.
- Detect and annotate any misinterpretations or hallucinations (such as assuming extra cards or misidentifying a flush).

Output Requirements:
- Provide a revised version of the math and board analysis sections with corrections applied.
- Include a brief summary of the changes you made.
- If any targeted section (mathematical computations or board analysis) is missing, explicitly mention it in your response.

Game State: {gamestate}
Optimal Action: {optimal_action}
Proposed Reasoning: {proposed_reasoning}
    """.format(gamestate=gamestate, optimal_action=optimal_action, proposed_reasoning=proposed_reasoning)
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


def prompt_meta_verifier(gamestate, optimal_action, proposed_reasoning, math_board_output, range_estimation_output):
    prompt = """
You are an expert poker strategist tasked with a comprehensive review of the chain-of-thought explanation for a poker scenario. In this review, you have access to multiple sources:
1. The original proposed chain-of-thought explanation.
2. The revised math and board analysis output (see below).
3. The revised opponent range estimation output (see below).

Your objectives are:

Evaluate Revisions:
- Determine if the corrections make sense, if there are any.

Overall Coherence and Accuracy:
- Integrate the corrections from the math/board analysis and opponent range estimation outputs into a final chain-of-thought explanation.
- Ensure that every part of the final explanation is logically consistent and factually correct according to standard poker strategy.
- Verify that all mathematical computations, board analyses, and opponent range analyses are valid and coherently merged.

Hallucination and Inconsistency Detection:
- Identify any remaining hallucinations, misrepresentations, or logical inconsistencies in the combined explanation.
- If there are any issues, provide detailed annotations and suggest further corrections if necessary.

Removal of Optimal Action References:
- Ensure that the final verified reasoning (chain-of-thought explanation) does not include any reference to the optimal action. Remove any such details so that the reasoning stands alone.

Final Output Requirements:
- Present a fully revised and integrated chain-of-thought explanation that is self-contained, free of any optimal action details, and adheres to strong game theory optimal poker concepts.

Below are the inputs:

Game State: {gamestate}
Optimal Action: {optimal_action}
Original Proposed Reasoning: {proposed_reasoning}

Math and Board Analysis Revision:
{math_board_output}

Opponent Range Estimation Revision:
{range_estimation_output}

Please proceed step by step in your verification.
    """.format(
        gamestate=gamestate,
        optimal_action=optimal_action,
        proposed_reasoning=proposed_reasoning,
        math_board_output=math_board_output,
        range_estimation_output=range_estimation_output
    )
    return prompt