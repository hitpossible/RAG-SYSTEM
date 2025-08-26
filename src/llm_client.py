import ollama
from typing import List, Dict, Any

class LlamaClient:
    def __init__(self, model_name: str = "llama3.2:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)
        
        # Check if model is available
        try:
            models = self.client.list()
            available_models = [model['model'] for model in models['models']]
            
            if model_name not in available_models:
                print(f"Model {model_name} not found. Available models: {available_models}")
                print(f"Please run: ollama pull {model_name}")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")

    
    def generate_response(self, prompt: str, context: List[Dict[str, Any]] = None, history: List[dict] = None) -> str:
        """Generate response using Llama model with improved RAG system prompt"""

        # Build context from retrieved documents
        context_text = ""
        if context:
            context_text = "\n\n".join([
                f"{doc['content']}" 
                for i, doc in enumerate(context)
            ])

        # --- Create improved system prompt ---
        if context_text.strip():
            system_prompt = (
                "You are a helpful assistant that provides accurate and comprehensive answers. "
                "You have been provided with relevant context documents to help answer the user's question. "
                
                "Guidelines for your response:\n"
                "1. PRIORITIZE information from the provided context when it's relevant and accurate\n"
                "2. You may supplement with your general knowledge when it adds value or clarifies concepts\n"
                "3. CLEARLY distinguish between information from context vs. your general knowledge\n"
                "4. If context information conflicts with your knowledge, acknowledge the discrepancy\n"
                "5. Cite specific sources when using context information\n"
                "6. If the context is insufficient for a complete answer, clearly state what's missing\n"
                "7. Provide reasoning and step-by-step explanations for complex questions\n"
                "8. Be concise but thorough - avoid unnecessary repetition"
            )
            
            user_prompt = f"""Context Documents:
    {context_text}

    ---
    Question: {prompt}

    Please provide a comprehensive answer using the context above as your primary source, supplemented with relevant general knowledge where appropriate. Clearly indicate when you're using context vs. general knowledge."""

        else:
            system_prompt = (
                "You are a helpful assistant that provides accurate and comprehensive answers "
                "based on your knowledge. No external context has been provided for this query. "
                
                "Guidelines:\n"
                "1. Be accurate and well-reasoned in your responses\n"
                "2. Provide step-by-step explanations for complex topics\n"
                "3. Acknowledge when you're uncertain about information\n"
                "4. Be concise but thorough\n"
                "5. Use examples to clarify concepts when helpful"
            )
            
            user_prompt = f"""Question: {prompt}

    Please provide the best possible answer based on your knowledge. If you're uncertain about any information, please indicate this clearly."""

        # --- Build messages ---
        messages = [{'role': 'system', 'content': system_prompt}]
        if history:
            # Optionally limit history length to prevent context overflow
            recent_history = history[-10:]  # Keep last 10 exchanges
            messages.extend(recent_history)
        messages.append({'role': 'user', 'content': user_prompt})

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': 0.7,  # Consider lowering to 0.3-0.5 for more factual responses
                    'top_p': 0.9,
                    'max_tokens': 512,
                    'repeat_penalty': 1.1  # Reduce repetition
                }
            )

            return response['message']['content']

        except Exception as e:
            return f"Error generating response: {e}"


    # Additional helper function for better context formatting
    def format_context_with_metadata(self, context: List[Dict[str, Any]]) -> str:
        """Format context with better metadata handling"""
        if not context:
            return ""
        
        formatted_parts = []
        for i, doc in enumerate(context):
            source_info = doc.get('source', 'Unknown Source')
            score = doc.get('score', 'N/A')
            content = doc.get('content', '')
            
            part = f"Document {i+1} (Relevance: {score}):\nSource: {source_info}\nContent: {content}"
            formatted_parts.append(part)
        
        return "\n\n" + "="*50 + "\n\n".join(formatted_parts)
