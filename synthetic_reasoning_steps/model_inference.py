#!/usr/bin/env python3
"""
Script to run inference with a PEFT fine-tuned model (LoRA).

Usage:
    python synthetic_reasoning_steps/model_inference.py \
        --base_model_name "meta-llama/Llama-3.1-8B-Instruct" \
        --adapter_path "synthetic_reasoning_steps/output/trained_model_flash_preview_04_17/" \
        --prompt "Your input prompt here" \
        [--max_new_tokens 250] \
        [--temperature 0.7] \
        [--gpu_id 0]

Example:
    python synthetic_reasoning_steps/model_inference.py \
        --base_model_name "meta-llama/Llama-3.1-8B-Instruct" \
        --adapter_path "synthetic_reasoning_steps/output/trained_model_flash_preview_04_17/" \
        --prompt "You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision. \n\nHere is a game summary: \n\nThe small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips. \nThe player positions involved in this game are UTG, HJ, CO, BTN, SB, BB.\nIn this hand, your position is BB (You are in position against SB), and your holding is [Eight of Heart and Eight of Club].\nBefore the flop, SB raise 3.0 chips, and BB call. Assume that all other players that is not mentioned folded.\nThe flop comes Ten Of Diamond, Six Of Heart, and Four Of Heart, then SB check, and BB check.\nThe turn comes King Of Heart, then SB check.\n\n\nYou currently have Flush. Your hand strength is Flush, King-high.\n\nNow it is your turn to make a move. \nTo remind you, your holding is [Eight of Heart and Eight of Club], the current pot size is 6 chips.\n\nDecide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer. \nYour optimal action is:" \
        --max_new_tokens 2000 \
        --temperature 0.7 \
        --gpu_id 0  
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging
import sys # Added for sys.exit

# Added import for helper function
from helper_functions import find_local_model_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with a PEFT fine-tuned model.')
    parser.add_argument('--base_model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help='Base model name or path (e.g., "meta-llama/Llama-3.1-8B-Instruct")')
    parser.add_argument('--adapter_path', type=str, required=True,
                       help='Path to the trained LoRA adapter directory.')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Input prompt for the model.')
    parser.add_argument('--max_new_tokens', type=int, default=250,
                       help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature.')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling top-p.')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling.')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='Specify the GPU ID to use (e.g., 0). Defaults to auto device map.')
    parser.add_argument('--merge_adapter', action='store_true',
                       help='Merge the adapter into the base model before inference (requires more memory).')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # --- Resolve Local Model Path ---
    logger.info(f"Attempting to find local path for base model: {args.base_model_name}")
    local_base_model_path = find_local_model_path(args.base_model_name)

    if local_base_model_path is None:
        logger.error(f"Base model '{args.base_model_name}' not found locally. Please ensure the model exists in the expected shared directory or cache.")
        logger.error("To prevent accidental downloads, this script will now exit.")
        sys.exit(1) # Exit if model not found locally
    else:
        logger.info(f"Using local base model path: {local_base_model_path}")

    # --- GPU Setup ---
    if args.gpu_id is not None:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count():
                 device_map = {"": f"cuda:{args.gpu_id}"} # Pin to specific GPU
                 logger.info(f"Using specified GPU: {args.gpu_id}")
            else:
                 raise ValueError(f"GPU {args.gpu_id} not available or invalid.")
        except Exception as e:
             logger.warning(f"Could not set GPU {args.gpu_id}: {e}. Using device_map='auto'.")
             device_map = "auto"
    else:
        device_map = "auto" # Auto-distribute across available devices
        logger.info("Using device_map='auto'.")

    # --- Load Model and Tokenizer ---
    logger.info(f"Loading base model from local path: {local_base_model_path}")

    # Quantization config (optional, adjust based on training setup and available VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        local_base_model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Set padding side to left for decoder-only models like Llama
        tokenizer.padding_side = "left"
        logger.info("Set pad_token to eos_token and padding_side to left.")

    logger.info(f"Loading LoRA adapter from: {args.adapter_path}")
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    if args.merge_adapter:
        logger.info("Merging adapter into base model...")
        try:
            model = model.merge_and_unload()
            logger.info("Adapter merged successfully.")
        except Exception as e:
            logger.warning(f"Could not merge adapter: {e}. Proceeding without merging.")

    model.eval() # Set model to evaluation mode

    # --- Prepare Input ---
    logger.info("Preparing input prompt...")
    # Format the prompt using the chat template appropriate for the base model
    # Assuming an instruction-following model like Llama-3-Instruct
    messages = [
        {"role": "user", "content": args.prompt}
        # Add system prompt if needed/used during training:
        # {"role": "system", "content": "You are a helpful poker assistant."},
    ]

    # Make sure model and inputs are on the same device if not using device_map="auto"
    # and not merging adapters. If device_map is 'auto' or adapter is merged,
    # the PeftModel/Transformers model handles device placement.
    if device_map != "auto" and not args.merge_adapter:
        target_device = f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu"
        logger.info(f"Moving model and inputs to {target_device}")
        model.to(target_device)
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, # Add the prompt structure for the assistant's reply
            return_tensors="pt"
        ).to(target_device)
    else:
        # Device placement is handled by device_map or merge_and_unload
         input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device) # Use model.device when device_map is used or model is merged


    # --- Generate Response ---
    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id
    }

    logger.info(f"Generating response with params: {generation_params}")
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model.generate(input_ids, **generation_params)

    # Decode the generated tokens, skipping the prompt tokens
    response_tokens = outputs[0][input_ids.shape[-1]:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

    print("\n" + "="*20 + " Model Response " + "="*20)
    print(response_text)
    print("="*56)

if __name__ == "__main__":
    main()
