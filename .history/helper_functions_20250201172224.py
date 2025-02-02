import torch
from openai import OpenAI
from collections import defaultdict

## global variables
_tokenizer = None
_openai_model_name = 'gpt-4o'
_client = None
_model = defaultdict(lambda: None)

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

def load_vllm_model(model_name="meta-llama/meta-llama-3.1-70b-instruct"):
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