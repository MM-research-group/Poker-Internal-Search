import os
from google import genai

class ApiKeyManager:
    """Manages multiple API keys and rotates them when rate limits are hit."""
    
    def __init__(self, key_prefix="GEMINI_API_KEY", max_keys=15):
        self.keys = []
        self.current_index = 0
        
        # Load all available API keys
        for i in range(1, max_keys + 1):
            key_name = f"{key_prefix}_{i}" if i > 1 else key_prefix
            key = os.environ.get(key_name)
            if key:
                self.keys.append(key)
        
        if not self.keys:
            raise ValueError("No API keys found in environment variables")
        
        print(f"Loaded {len(self.keys)} API keys")
    
    def get_current_key(self):
        """Returns the currently active API key."""
        return self.keys[self.current_index]
    
    def rotate_key(self):
        """Rotates to the next available API key."""
        self.current_index = (self.current_index + 1) % len(self.keys)
        return self.keys[self.current_index]
    
    def get_client(self):
        """Returns a client initialized with the current API key."""
        return genai.Client(api_key=self.get_current_key())
