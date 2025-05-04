import json
import tiktoken
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────

# 1) path to your JSON file
fn = Path("/home/xuandongz/mnt/MMteam_cs194/Poker-Internal-Search/synthetic_reasoning_steps/results/DeepSeek-R1/50_sample.json")

enc = tiktoken.get_encoding("cl100k_base")

# 3) your per-million-token rates:
RATE_INPUT_PER_1M  = 0.55  
RATE_OUTPUT_PER_1M = 2.19  

# ─── HELPERS ───────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# ─── LOAD & COUNT ───────────────────────────────────────────────────────────────

data = json.loads(fn.read_text())

# we'll accumulate totals across all calls
total_input_tokens  = 0
total_output_tokens = 0
n_calls = 0

for record in data:
    # assume each of these keys corresponds to one API call
    for call_name in ("proposer_output",
                      "board_output",
                      "range_estimation_output",
                      "meta_output"):
        call = record[call_name]
        # define what you treat as “input” vs “output”
        # e.g. maybe gamestate + optimal_action were sent *in*, and
        # reasoning_content + final_response came *out*.
        # adjust these lists to match your actual prompt design!
        input_strings  = [ call["gamestate"], call["optimal_action"] ]
        output_strings = [ call["reasoning_content"], call["final_response"] ]

        toks_in  = sum(count_tokens(s) for s in input_strings)
        toks_out = sum(count_tokens(s) for s in output_strings)

        total_input_tokens  += toks_in
        total_output_tokens += toks_out
        n_calls += 1

# ─── AVERAGE & COST ────────────────────────────────────────────────────────────

avg_in  = total_input_tokens  / n_calls
avg_out = total_output_tokens / n_calls

cost_per_call = (avg_in  / 1_000_000) * RATE_INPUT_PER_1M \
               + (avg_out / 1_000_000) * RATE_OUTPUT_PER_1M

print(f"total calls:      {n_calls}")
print(f"avg input tokens:  {avg_in:.1f}")
print(f"avg output tokens: {avg_out:.1f}")
print(f"≈ cost per call:   ${cost_per_call:.6f}")
print(f"≈ cost total:      ${cost_per_call * n_calls:.2f}")
