"""
Usage:
    python synthetic_reasoning_steps/sft.py 
        --model_name MODEL_NAME 
        --traindata_path DATASET_PATH 
        --output_dir OUTPUT_PATH 
        [--batch_size BATCH_SIZE] 
        [--learning_rate LEARNING_RATE]
        [--num_epochs NUM_EPOCHS]
        [--max_length MAX_LENGTH]
        [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
        [--lora_r LORA_R]
        [--lora_alpha LORA_ALPHA]
        [--lora_dropout LORA_DROPOUT]

This script implements a simple PEFT SFT pipeline using LoRA to finetune LLMs.
"""

import os
import sys
import argparse
import json
import logging
import torch
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from tqdm import tqdm
from dotenv import load_dotenv

# Import helper functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_reasoning_steps.helper_functions import (
    setup_hf_env, find_local_model_path, load_dataset
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune a language model using PEFT (LoRA).')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help='Base model name to fine-tune')
    parser.add_argument('--traindata_path', type=str, required=True,
                       help='Path to JSON training dataset file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to save the fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Number of steps for gradient accumulation')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    return parser.parse_args()

def prepare_dataset(dataset_path, tokenizer, max_length):
    """Prepare and tokenize dataset for training.
    
    Args:
        dataset_path (str): Path to the training dataset
        tokenizer: Tokenizer to process the texts
        max_length (int): Maximum sequence length
        
    Returns:
        Dataset: HuggingFace Dataset ready for training
    """
    # Load dataset from JSON file
    raw_data = load_dataset(dataset_path)
    
    # Convert to format expected by datasets library
    processed_data = []
    for item in raw_data:
        # Dataset is expected to contain prompts and completions
        # Format: {"input": "user prompt", "output": "assistant response"}
        if "input" in item and "output" in item:
            prompt = item["input"]
            completion = item["output"]
            
            # Format into chat template
            processed_data.append({
                "text": tokenizer.apply_chat_template([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ], tokenize=False)
            })
    
    dataset = Dataset.from_list(processed_data)
    
    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    logger.info(f"Dataset processed and tokenized: {len(tokenized_dataset)} examples")
    return tokenized_dataset

def load_base_model(model_name):
    """Load base model and tokenizer.
    
    Args:
        model_name (str): Base model name or path
        
    Returns:
        tuple: (model, tokenizer)
    """
    # First check if model is available locally
    local_path = find_local_model_path(model_name)
    model_path = local_path if local_path else model_name
    
    # Configure quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Make sure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 4-bit precision
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def train_model(args, model, tokenizer, train_dataset):
    """Fine-tune model using LoRA.
    
    Args:
        args: Command line arguments
        model: Base model to fine-tune
        tokenizer: Tokenizer for the model
        train_dataset: Processed training dataset
        
    Returns:
        Fine-tuned model
    """
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Common target modules for Llama models
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=3,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=False,  # Use fp16 instead as it's more widely supported
        fp16=True,
    )
    
    # Create data collator for language modeling
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
    
    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model and tokenizer
    logger.info(f"Saving fine-tuned model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    return model

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup HF environment
    setup_hf_env()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base model and tokenizer
    logger.info(f"Loading base model: {args.model_name}")
    model, tokenizer = load_base_model(args.model_name)
    
    # Prepare dataset
    logger.info(f"Processing training data from: {args.traindata_path}")
    train_dataset = prepare_dataset(args.traindata_path, tokenizer, args.max_length)
    
    # Train model
    trained_model = train_model(args, model, tokenizer, train_dataset)
    
    logger.info(f"Fine-tuning completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
