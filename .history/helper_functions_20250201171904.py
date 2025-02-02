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