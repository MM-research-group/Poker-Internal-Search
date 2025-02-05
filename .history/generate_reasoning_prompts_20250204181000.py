def prompt_proposer(gamestate, optimal_action):
    prompt = """
You are a professional poker player known for your deep strategic insight. Given the following game state and knowing that the solver has determined the optimal action to be {optimal_action}, please provide a detailed chain-of-thought explanation for why this action is the best choice. In your explanation, consider the following:
- What is my opponent's range here? Is their range capped or uncapped?
- What is our hand? (value, showdown value, bluff)
- Given the opponent's range, what can we do about it?
- Risk management, pot odds, stack sizes (when necessary)
- Future street considerations and board texture

Only use factors that makes sense given the current gamestate. Break down your reasoning step by step, and ensure that each part of your explanation logically follows from the game state and known poker strategies. Be careful to avoid speculative statements that are not grounded in standard poker theory.

Game State: {gamestate}
Optimal Action: {optimal_action}

Please provide your output in a structured JSON format with the following keys:
- "chain_of_thought": Your detailed step-by-step explanation.
- "summary_of_steps": A brief summary of the main points in your explanation.
If any part is not applicable, please use an empty string for that key.
    """.format(gamestate=gamestate, optimal_action=optimal_action)
    return prompt


def prompt_math_board_verifier(gamestate, optimal_action, proposed_reasoning):
    prompt = """
You are a dedicated expert in verifying mathematical reasoning and board analysis in poker scenarios.
Below is the reasoning steps that correspond to the gamestate provided below, which should include both mathematical computations and board analysis. (If one or both are missing, please note that explicitly in your output.)

Your tasks are to verify the following areas within the reasoning steps:

Mathematical Verification:
- Identify every instance where mathematical computations or numerical reasoning is applied (e.g., probability calculations, pot odds, percentage computations).
- Verify that all arithmetic is correct and that any probability or risk assessments follow standard poker math.
- Flag and explain any miscalculations or inconsistencies.

Board Analysis Verification:
- Review all sections where the board is analyzed, particularly noting card counts, suit distributions, potential flushes, straight draws, or other spatial/card pattern considerations.
- Ensure that the description of the board (e.g., number of spades, hearts, etc.) is consistent with the actual game state.
- Detect and annotate any misinterpretations or hallucinations such as assuming extra cards or misidentifying a flush.

Output Requirements:
- Provide a revised version of the math and board analysis sections with all corrections applied.
- Include a brief summary of the changes you made.
- If any targeted section (mathematical computations or board analysis) is missing, explicitly mention it in your output.

Please provide your response in a structured JSON format with the following keys:
- "revised_text": The corrected math and board analysis portions.
- "summary_of_changes": A summary of corrections and annotations.
- "missing_sections": A list of any sections that are missing (or an empty list if none are missing).

Game State: {gamestate}
Optimal Action: {optimal_action}
Proposed Reasoning: {proposed_reasoning}
    """.format(gamestate=gamestate, optimal_action=optimal_action, proposed_reasoning=proposed_reasoning)
    return prompt

def prompt_range_estimation_verifier(gamestate, optimal_action, proposed_reasoning):
    prompt = """
You are an expert in opponent range estimation in poker scenarios. Your task is to evaluate the portion of the chain-of-thought reasoning steps below that discusses the opponent's potential holdings based on the provided game state and betting actions.

Your tasks are:

Opponent Range Analysis Verification:
- Analyze the reasoning that identifies the opponent's possible holdings, taking into account their pre-flop, flop, and subsequent actions.
- Evaluate whether the explanation logically categorizes the opponent’s range into segments such as made hands, draws, and potential bluffs.
- Assess if the explanation provides an accurate estimation of the proportion or likelihood of various holdings (e.g., the balance between draws and made hands).
- Flag any inaccuracies, misinterpretations, or omissions in the opponent range analysis and provide appropriate corrections.

Output Requirements:
- Provide a revised version of the opponent range estimation section with all necessary corrections applied.
- Include a brief summary of the changes you made.
- If the explanation is missing an analysis of opponent range, explicitly mention this in your output.

Please provide your response in a structured JSON format with the following keys:
- "revised_text": The corrected opponent range estimation analysis.
- "summary_of_changes": A summary of the modifications and corrections made.
- "missing_sections": A list of any missing sections (or an empty list if none are missing).

Game State: {gamestate}
Optimal Action: {optimal_action}
Proposed Reasoning: {proposed_reasoning}
    """.format(gamestate=gamestate, optimal_action=optimal_action, proposed_reasoning=proposed_reasoning)
    return prompt


def prompt_meta_verifier(gamestate, optimal_action, proposed_reasoning, math_board_output, range_estimation_output):
    prompt = """
You are an expert poker strategist tasked with a comprehensive review of the chain-of-thought explanation for a poker scenario. In this review, you have access to multiple sources:
1. The original proposed chain-of-thought explanation.
2. The revised math and board analysis output, provided below.
3. The revised opponent range estimation output, provided below.

Your objectives are:

Overall Coherence and Accuracy:
- Integrate any corrections from the math/board analysis and opponent range estimation outputs into a final chain-of-thought explanation.
- Ensure that every part of the final explanation is logically consistent and factually correct according to standard poker strategy.
- Verify that all mathematical computations, board analyses, and opponent range analyses are valid and coherently merged with the rest of the reasoning.

Hallucination and Inconsistency Detection:
- Identify any remaining hallucinations, misrepresentations, or logical inconsistencies in the combined explanation.
- Provide detailed annotations on any issues found and suggest further corrections if necessary.

Removal of Optimal Action References:
- It is imperative that the final verified explanation does not include any reference to, or information about, the optimal action. Remove any such details so that the reasoning stands alone.

Final Output Requirements:
- Present a fully revised and integrated chain-of-thought explanation that is self-contained, free of any optimal action details, and adheres to poker best practices.
- Provide a brief summary of the changes and corrections made during your review.
- If any sections are still missing, explicitly note them.

Please provide your response in a structured JSON format with the following keys:
- "final_chain_of_thought": The fully revised and integrated chain-of-thought explanation.
- "summary_of_changes": A summary of the modifications and corrections applied.
- "missing_sections": A list of any missing sections (or an empty list if none are missing).

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