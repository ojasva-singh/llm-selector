import os
import yaml
import time
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import anthropic
import openai
from google import generativeai as genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class QueryType(Enum):
    REASONING = "reasoning"
    FAST = "fast"
    COST_OPTIMIZED = "cost_optimized"


@dataclass
class ModelConfig:
    name: str
    provider: str
    api_key_env: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    max_tokens: int


class QueryAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.reasoning_keywords = config['routing']['reasoning_keywords']
        self.fast_keywords = config['routing']['fast_keywords']
        self.long_threshold = config['routing']['long_response_threshold']
        self.complex_threshold = config['routing']['complex_query_length']
    
    def analyze(self, query: str) -> Tuple[QueryType, Dict]:
        query_lower = query.lower()
        word_count = len(query.split())
        
        analysis = {
            'query_length': word_count,
            'has_reasoning_keywords': any(keyword in query_lower for keyword in self.reasoning_keywords),
            'has_fast_keywords': any(keyword in query_lower for keyword in self.fast_keywords),
            'is_complex': word_count > self.complex_threshold,
            'expects_long_response': self._expects_long_response(query_lower)
        }
        
        if analysis['has_reasoning_keywords'] or analysis['is_complex']:
            return QueryType.REASONING, analysis
        elif analysis['expects_long_response']:
            return QueryType.COST_OPTIMIZED, analysis
        else:
            return QueryType.FAST, analysis
    
    def _expects_long_response(self, query: str) -> bool:
        long_indicators = ['detailed', 'comprehensive', 'explain in detail', 'write', 'essay', 'article', 'tutorial']
        return any(indicator in query for indicator in long_indicators)


class LLMRouter:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.analyzer = QueryAnalyzer(self.config)
        self.retry_config = self.config['retry']
        self._init_clients()
    
    def _init_clients(self):
        self.clients = {}
        
        if os.getenv('ANTHROPIC_API_KEY'):
            self.clients['anthropic'] = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        if os.getenv('OPENAI_API_KEY'):
            self.clients['openai'] = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        if os.getenv('GOOGLE_API_KEY'):
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.clients['google'] = genai
        
        if os.getenv('GROQ_API_KEY'):
            self.clients['groq'] = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    def route_query(self, query: str, verbose: bool = True) -> Dict:
        query_type, analysis = self.analyzer.analyze(query)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Query Type: {query_type.value}")
            print(f"Analysis: {analysis}")
            print(f"{'='*60}\n")
        
        model_category = self.config['models'][query_type.value]
        primary_model = ModelConfig(**model_category['primary'])
        fallback_models = [ModelConfig(**m) for m in model_category['fallbacks']]
        all_models = [primary_model] + fallback_models
        
        for idx, model in enumerate(all_models):
            is_primary = (idx == 0)
            model_type = "PRIMARY" if is_primary else f"FALLBACK {idx}"
            
            if verbose:
                print(f"🔄 {model_type}: {model.name} ({model.provider})")
            
            try:
                response = self._call_model_with_retry(model, query)
                
                if verbose:
                    print(f"✅ Success with {model.name}\n")
                
                return {
                    'success': True,
                    'model_used': model.name,
                    'provider': model.provider,
                    'is_fallback': not is_primary,
                    'response': response,
                    'analysis': analysis
                }
            
            except Exception as e:
                if verbose:
                    print(f"❌ Failed: {str(e)[:100]}")
                    if idx < len(all_models) - 1:
                        print(f"   Falling back...\n")
                
                if idx == len(all_models) - 1:
                    return {
                        'success': False,
                        'error': f"All models failed. Last error: {str(e)}",
                        'analysis': analysis
                    }
    
    def _call_model_with_retry(self, model: ModelConfig, query: str) -> str:
        attempt = 0
        delay = self.retry_config['initial_delay']
        
        while attempt < self.retry_config['max_attempts']:
            try:
                return self._call_model(model, query)
            except Exception as e:
                attempt += 1
                if attempt >= self.retry_config['max_attempts']:
                    raise
                time.sleep(delay)
                delay = min(delay * self.retry_config['exponential_base'], self.retry_config['max_delay'])
    
    def _call_model(self, model: ModelConfig, query: str) -> str:
        provider = model.provider
        
        if provider == 'anthropic':
            client = self.clients.get('anthropic')
            if not client:
                raise Exception("Anthropic API key not configured")
            message = client.messages.create(
                model=model.name,
                max_tokens=model.max_tokens,
                messages=[{"role": "user", "content": query}]
            )
            return message.content[0].text
        
        elif provider == 'openai':
            client = self.clients.get('openai')
            if not client:
                raise Exception("OpenAI API key not configured")
            response = client.chat.completions.create(
                model=model.name,
                messages=[{"role": "user", "content": query}],
                max_tokens=model.max_tokens
            )
            return response.choices[0].message.content
        
        elif provider == 'google':
            client = self.clients.get('google')
            if not client:
                raise Exception("Google API key not configured")
            model_obj = client.GenerativeModel(model.name)
            response = model_obj.generate_content(query)
            return response.text
        
        elif provider == 'groq':
            client = self.clients.get('groq')
            if not client:
                raise Exception("Groq API key not configured")
            response = client.chat.completions.create(
                model=model.name,
                messages=[{"role": "user", "content": query}],
                max_tokens=model.max_tokens
            )
            return response.choices[0].message.content
        
        else:
            raise Exception(f"Unknown provider: {provider}")


def main():
    router = LLMRouter()
    
    print("\n" + "="*60)
    print("🤖 LLM Router")
    print("="*60)
    
    while True:
        print("\nEnter query (quit/q/exit to exit):")
        query = input("> ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        result = router.route_query(query, verbose=True)
        
        if result['success']:
            print(f"\n{'='*60}")
            print(f"Response ({result['model_used']}):")
            print(f"{'='*60}")
            print(result['response'])
            print(f"{'='*60}\n")
        else:
            print(f"\n❌ Error: {result['error']}\n")


if __name__ == "__main__":
    main()
