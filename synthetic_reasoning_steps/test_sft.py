"""
Test script for the SFT pipeline.
This runs a mini training job with minimal epochs and steps to verify functionality.

Usage:
    python test_sft.py [--gpu_ids GPU_IDS]
"""

import os
import sys
import torch
import logging
import tempfile
import argparse
from transformers import set_seed

# Add the parent directory to the path to import helper_functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_reasoning_steps.sft import (
    load_base_model,
    prepare_dataset,
    train_model,
    setup_hf_env,
    setup_gpu_environment
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the SFT pipeline with minimal configuration.')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='GPU ID to use (e.g., "0"). Only the first GPU will be used if multiple are specified.')
    return parser.parse_args()

def test_training_pipeline(gpu_ids=None):
    """Test the entire training pipeline with minimal settings.
    
    Args:
        gpu_ids (str, optional): GPU ID to use. Only the first will be used.
        
    Returns:
        bool: Whether the test was successful
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup GPU environment
    setup_gpu_environment(gpu_ids)
    
    # Setup HF environment
    setup_hf_env()
    
    # Use a small model for testing if you can't access Llama-3.1
    # Try using a smaller model first that doesn't require HF token
    try:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        logger.info(f"Testing with model: {model_name}")
        model, tokenizer = load_base_model(model_name)
    except Exception as e:
        logger.warning(f"Error loading Llama model: {e}")
        logger.info("Falling back to a smaller open model: TinyLlama")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        model, tokenizer = load_base_model(model_name)
    
    # Path to test data
    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.json")
    if not os.path.exists(test_data_path):
        logger.error(f"Test data not found at {test_data_path}")
        sys.exit(1)
    
    # Create a temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Using temporary directory for output: {temp_dir}")
        
        try:
            # Process test dataset - with lower max_length for faster testing
            train_dataset = prepare_dataset(test_data_path, tokenizer, max_length=256)
            
            # Create a minimal args object for training
            class Args:
                def __init__(self):
                    self.output_dir = temp_dir
                    self.batch_size = 1
                    self.learning_rate = 2e-4
                    self.num_epochs = 1  # Minimal epochs for testing
                    self.gradient_accumulation_steps = 1
                    self.lora_r = 4     # Smaller LoRA rank for testing
                    self.lora_alpha = 8
                    self.lora_dropout = 0.05
                    self.max_length = 256
                    self.gpu_ids = gpu_ids
            
            args = Args()
            
            # Run minimal training
            logger.info("Starting minimal training run...")
            model = train_model(args, model, tokenizer, train_dataset)
            logger.info("Training completed successfully!")
            
            # Verify that the model and adapter config were saved
            adapter_config_path = os.path.join(temp_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                logger.info("✅ Adapter config file created successfully.")
            else:
                logger.warning("❌ Adapter config file not found.")
            
            # Test inference with the adapted model
            logger.info("Testing inference with the trained model...")
            test_prompt = "What is parameter-efficient fine-tuning?"
            
            # Format the prompt using the chat template
            formatted_prompt = tokenizer.apply_chat_template([
                {"role": "user", "content": test_prompt}
            ], tokenize=False)
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
            
            with torch.no_grad():
                # Generate with a small max_length to keep it quick
                output_ids = model.generate(**inputs, max_length=100, num_return_sequences=1)
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            logger.info(f"Test Prompt: {test_prompt}")
            logger.info(f"Model Output: {output_text}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training test: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    logger.info("Starting SFT pipeline test...")
    success = test_training_pipeline(args.gpu_ids)
    
    if success:
        logger.info("✅ SFT pipeline test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ SFT pipeline test failed!")
        sys.exit(1) 