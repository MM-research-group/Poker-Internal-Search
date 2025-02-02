from collections import defaultdict

## global variables
_tokenizer = None
_openai_model_name = 'gpt-4o'
_client = None
_model = defaultdict(lambda: None)

