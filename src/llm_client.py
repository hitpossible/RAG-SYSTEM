import ollama
from typing import List, Dict, Any

class LlamaClient:
    def __init__(self, model_name: str = "llama3.2:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)
        
        # Check if model is available
        try:
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            print(available_models)
            if model_name not in available_models:
                print(f"Model {model_name} not found. Available models: {available_models}")
                print(f"Please run: ollama pull {model_name}")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
    
    def generate_response(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """Generate response using Llama model"""
        
        # Build context from retrieved documents
        context_text = ""
        if context:
            context_text = "\n\n".join([doc['content'] for doc in context])
    
        # สร้าง system prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
        If the context doesn't contain relevant information, say so clearly.
        Always be accurate and cite your sources when possible."""

        # สร้าง user prompt โดยจะเพิ่ม context ถ้ามี
        if context_text:
            user_prompt = f"""Context: {context_text}
            Question: {prompt}
            Please provide a comprehensive answer based on the context above."""
        else:
            user_prompt = f"""Question: {prompt}
            Please provide a comprehensive answer."""

        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 2000
                }
            )
            
            return response['message']['content']
        
        except Exception as e:
            return f"Error generating response: {e}"