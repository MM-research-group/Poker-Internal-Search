'''
Usage:
    python synthetic_reasoning_steps/evaluate.py [--model MODEL_NAME] [--prompt PROMPT] [--temp TEMPERATURE] [--max_tokens MAX_TOKENS] [--test-gemini]
'''

import sys
import os
import argparse
from vllm import SamplingParams
from helper_functions import load_model, setup_hf_env
from google import genai
from dotenv import load_dotenv

def test_gemini_api():
    """Test the Gemini API connection and functionality."""
    print("\n--- Testing Gemini API Configuration ---")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if Gemini API key exists
    if 'GEMINI_API_KEY' not in os.environ:
        print("ERROR: No GEMINI_API_KEY found in environment.")
        print("Please add your Gemini API key to your .env file as GEMINI_API_KEY=your_key_here")
        return False
    
    # Configure Gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    genai_client = genai.Client(api_key=api_key)
    
    print("✓ Gemini API key found")
    
    # Test prompt
    test_prompt = "What are three interesting facts about neural networks? Keep it short."
    
    try:
        # Generate response
        print(f"Sending test prompt to Gemini API: '{test_prompt}'")
        
        # Send the prompt to Gemini using the updated API
        response = genai_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=test_prompt,
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=100
        )
        
        # Display output
        print("\nGemini API test successful!")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
        
        print("✓ Gemini API is configured correctly and working!")
        print("You can use the --model gemini-1.5-pro option with evaluate.py")
        
        # Test successful
        return True
        
    except Exception as e:
        print(f"\nERROR with Gemini API: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your API key is correct")
        print("2. Ensure you have quota/credits available for Gemini API")
        print("3. Check your internet connection")
        print("4. Verify the model name (gemini-1.5-pro) is available to you")
        return False

def setup_environment():
    """Set up the Hugging Face environment and validate token exists."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for HF_TOKEN
    hf_home = setup_hf_env()
    if not os.getenv('HF_TOKEN'):
        print("ERROR: No HF_TOKEN found in environment. Cannot access gated models.")
        sys.exit(1)
    
    # Initialize Gemini client if API key exists
    if 'GEMINI_API_KEY' in os.environ:
        # We'll initialize the client when needed
        pass
    else:
        print("WARNING: No GEMINI_API_KEY found in environment. Gemini models won't work.")
    
    return hf_home

def load_model_instance(model_name):
    """Load the specified model and return it."""
    print(f"Loading model: {model_name}")
    
    if "gemini" in model_name.lower():
        # For Gemini, return the model name since we'll use the API directly
        return model_name
    else:
        # For vLLM models (like LLaMA)
        return load_model(model_name)

def create_sampling_params(temperature=0.7, max_tokens=100, top_p=0.95):
    """Create and return sampling parameters."""
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

def generate_response(model, prompt, sampling_params, model_name):
    """Generate a response from the model."""
    print(f"Sending test prompt: '{prompt}'")
    
    # Handle different model types
    if "gemini" in model_name.lower():
        # For Gemini API
        try:
            # Set up API client
            api_key = os.environ.get("GEMINI_API_KEY")
            genai_client = genai.Client(api_key=api_key)
            
            # Generate content
            response = genai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                generation_config={
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                    "max_output_tokens": sampling_params.max_tokens,
                }
            )
            
            # Create a structure similar to vLLM output for consistent handling
            class GeminiOutput:
                def __init__(self, text):
                    self.text = text
            
            class GeminiOutputWrapper:
                def __init__(self, content):
                    self.outputs = [GeminiOutput(content)]
            
            return [GeminiOutputWrapper(response.text)]
            
        except Exception as e:
            print(f"ERROR with Gemini API: {e}")
            # Return error message as output to maintain consistent structure
            class ErrorOutput:
                def __init__(self, error):
                    self.text = f"Error: {error}"
            
            class ErrorOutputWrapper:
                def __init__(self, error):
                    self.outputs = [ErrorOutput(error)]
            
            return [ErrorOutputWrapper(str(e))]
    else:
        # For vLLM models
        return model.generate([prompt], sampling_params)

def display_output(outputs):
    """Display the model's output."""
    print("\nModel output:")
    print("-" * 50)
    for output in outputs:
        print(output.outputs[0].text)
    print("-" * 50)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a language model.')
    parser.add_argument('--model', type=str, default="meta-llama/meta-llama-3.1-8b-instruct",
                       help='Model name to evaluate (use "gemini-1.5-pro" for Gemini API)')
    parser.add_argument('--prompt', type=str,
                       default="Please explain what neural networks are in 3 sentences.",
                       help='Test prompt to send to the model')
    parser.add_argument('--temp', type=float, default=0.7,
                       help='Temperature for sampling')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Top-p sampling parameter')
    parser.add_argument('--test-gemini', action='store_true',
                       help='Run a test of the Gemini API configuration')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # If testing Gemini API, do just that and exit
    if args.test_gemini:
        test_gemini_api()
        return
    
    # Set up environment
    setup_environment()
    
    # Load model
    model = load_model_instance(args.model)
    
    # Create sampling parameters
    sampling_params = create_sampling_params(
        temperature=args.temp,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )
    
    # Generate response
    outputs = generate_response(model, args.prompt, sampling_params, args.model)
    
    # Display output
    display_output(outputs)
    
    print("Model test completed successfully!")

if __name__ == "__main__":
    main()
