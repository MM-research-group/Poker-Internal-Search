'''
Usage:
    python synthetic_reasoning_steps/evaluate.py
'''

import sys
from vllm import SamplingParams
from helper_functions import load_vllm_model

def main():
    model_name = "meta-llama/meta-llama-3.1-8b-instruct"
    print(f"Loading model: {model_name}")
    
    model = load_vllm_model(model_name)
    
    test_prompt = "Please explain what neural networks are in 3 sentences."
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=100,
        top_p=0.95,
    )
    
    print(f"Sending test prompt: '{test_prompt}'")
    outputs = model.generate([test_prompt], sampling_params)
    
    print("\nModel output:")
    print("-" * 50)
    for output in outputs:
        print(output.outputs[0].text)
    print("-" * 50)
    
    print("Model test completed successfully!")

if __name__ == "__main__":
    main()
