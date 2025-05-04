import torch
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from collections import defaultdict
import os
import json
from dotenv import load_dotenv

## global variables
_tokenizer = None
_openai_model_name = 'gpt-4o'
_client = None
_model = defaultdict(lambda: None)

def setup_hf_env():
    """Set up Hugging Face token from config file or .env file."""
    load_dotenv()
    
    if os.getenv('HF_TOKEN'):
        print("Using HF_TOKEN from environment variables")
        return os.getenv('HF_HOME')
    
    try:
        dir_of_this_script = os.path.dirname(os.path.abspath(__file__))
        path_to_config = os.path.join(dir_of_this_script, 'configs', 'config.json')
        
        if os.path.exists(path_to_config):
            with open(path_to_config, 'r') as config_file:
                config_data = json.load(config_file)
            
            if "HF_TOKEN" in config_data:
                os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
                print("Using HF_TOKEN from config file")
                return os.getenv('HF_HOME')
    except Exception as e:
        print(f"Error loading config: {e}")
    
    print("Warning: HF_TOKEN not found in environment or config file")
    return None

def load_model(model_name):
    """Generic function to either load a VLLM model or a client (e.g. OpenAI)."""
    if "gpt" in model_name:
        global _openai_model_name
        _openai_model_name = model_name
        return get_openai_client()
    else:
        return load_vllm_model(model_name)
    
def get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def load_vllm_model(model_name="meta-llama/meta-llama-3.1-8b-instruct"):
    global _model
    if _model[model_name] is None:
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)

        model = LLM(
            model_name,
            dtype=torch.float16,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=True,
            max_model_len=60_000
        )

        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        
        print(f"Model {model_name} loaded. Memory Allocated: {memory_allocated / (1024 ** 3):.2f} GB")
        print(f"Model {model_name} loaded. Memory Reserved: {memory_reserved / (1024 ** 3):.2f} GB")
        _model[model_name] = model
    else:
        model = _model[model_name]
    _ = initialize_tokenizer(model_name) # cache the tokenizer 
    return model

def initialize_tokenizer(model_name=None):
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer