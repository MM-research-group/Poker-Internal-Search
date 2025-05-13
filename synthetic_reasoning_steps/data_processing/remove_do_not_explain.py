import json

# --- Configuration ---
input_file_path = "/home/xuandong/mnt/poker/michael-poker-synthetic-reasoning-steps/synthetic_reasoning_steps/all_results_gemini-2.5-flash-preview-04-17/transformed_postflop_54785_synthetic_reasoning_steps_train_set.json"  
output_file_path = "/home/xuandong/mnt/poker/michael-poker-synthetic-reasoning-steps/synthetic_reasoning_steps/all_results_gemini-2.5-flash-preview-04-17/final_processed_postflop_54785_synthetic_reasoning_steps_train_set.json" 
string_to_remove = " Do not explain your answer. "
# ---------------------

# to run: 
# python -m synthetic_reasoning_steps.data_processing.remove_do_not_explain

def remove_do_not_explain_from_json(input_path: str, output_path: str):
    """
    Reads a JSON file, removes a specific substring from the 'input' field
    of each object, and writes the modified data to a new JSON file.

    Args:
        input_path: Path to the input JSON file.
        output_path: Path to the output JSON file.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
        return

    if not isinstance(data, list):
        print(f"Error: Expected a list of JSON objects in {input_path}")
        return

    modified_data = []
    for item in data:
        if isinstance(item, dict) and "input" in item and isinstance(item["input"], str):
            item["input"] = item["input"].replace(string_to_remove, "")
        # Keep the item even if it doesn't match the expected structure
        modified_data.append(item)


    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(modified_data, outfile, indent=2, ensure_ascii=False)
        print(f"Successfully processed file. Output saved to {output_path}")
    except IOError as e:
        print(f"Error writing to output file {output_path}: {e}")

if __name__ == "__main__":
    remove_do_not_explain_from_json(input_file_path, output_file_path)
