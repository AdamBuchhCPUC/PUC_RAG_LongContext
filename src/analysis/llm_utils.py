"""
LLM utilities for OpenAI API calls and cost calculation.
Handles reasoning models and standard models with proper error handling.
"""

import os
import openai
from typing import Dict, List, Tuple
import streamlit as st
from dotenv import load_dotenv
from .model_config import is_reasoning_model

# Load environment variables from .env file
load_dotenv()

# Set up your OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in .env file. Please create a .env file with your OpenAI API key.")


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """
    Calculate the estimated cost based on input and output tokens with separate pricing.
    Prices are per 1K tokens (current OpenAI pricing).
    """
    # OpenAI pricing per 1K tokens (input/output separate) - Corrected 2024
    input_pricing = {
        "gpt-3.5-turbo": 0.0010,    # $0.0010 per 1K tokens
        "gpt-4o-mini": 0.00015,     # $0.15 per 1M tokens = $0.00015 per 1K tokens
        "gpt-4o": 0.0025,           # $2.50 per 1M tokens = $0.0025 per 1K tokens
        "gpt-4.1-mini": 0.0004,     # $0.40 per 1M tokens = $0.0004 per 1K tokens
        "gpt-4.1": 0.002,           # $2.00 per 1M tokens = $0.002 per 1K tokens
        "o1-mini": 0.0011,         # $1.10 per 1M tokens = $0.0011 per 1K tokens
        "o1": 0.015,                # $15.00 per 1M tokens = $0.015 per 1K tokens
        "o3": 0.002,                # $2.00 per 1M tokens = $0.002 per 1K tokens
        "o3-pro": 0.02,             # $20.00 per 1M tokens = $0.02 per 1K tokens
        "o4-mini": 0.0011,          # $1.10 per 1M tokens = $0.0011 per 1K tokens
        "o3-mini": 0.0011,          # $1.10 per 1M tokens = $0.0011 per 1K tokens
        "gpt-5": 0.00125,           # $1.25 per 1M tokens = $0.00125 per 1K tokens
        "gpt-5-mini": 0.00025,      # $0.25 per 1M tokens = $0.00025 per 1K tokens
        "gpt-5-nano": 0.00005       # $0.05 per 1M tokens = $0.00005 per 1K tokens
    }
    
    output_pricing = {
        "gpt-3.5-turbo": 0.0020,    # $0.0020 per 1K tokens
        "gpt-4o-mini": 0.0006,      # $0.60 per 1M tokens = $0.0006 per 1K tokens
        "gpt-4o": 0.01,             # $10.00 per 1M tokens = $0.01 per 1K tokens
        "gpt-4.1-mini": 0.0016,     # $1.60 per 1M tokens = $0.0016 per 1K tokens
        "gpt-4.1": 0.008,           # $8.00 per 1M tokens = $0.008 per 1K tokens
        "o1-mini": 0.0044,          # $4.40 per 1M tokens = $0.0044 per 1K tokens
        "o1": 0.06,                 # $60.00 per 1M tokens = $0.06 per 1K tokens
        "o3": 0.008,                # $8.00 per 1M tokens = $0.008 per 1K tokens
        "o3-pro": 0.08,             # $80.00 per 1M tokens = $0.08 per 1K tokens
        "o4-mini": 0.0044,          # $4.40 per 1M tokens = $0.0044 per 1K tokens
        "o3-mini": 0.0044,          # $4.40 per 1M tokens = $0.0044 per 1K tokens
        "gpt-5": 0.01,              # $10.00 per 1M tokens = $0.01 per 1K tokens
        "gpt-5-mini": 0.002,        # $2.00 per 1M tokens = $0.002 per 1K tokens
        "gpt-5-nano": 0.0004        # $0.40 per 1M tokens = $0.0004 per 1K tokens
    }
    
    if model not in input_pricing or model not in output_pricing:
        return 0.0
    
    input_cost = (input_tokens / 1000) * input_pricing[model]
    output_cost = (output_tokens / 1000) * output_pricing[model]
    total_cost = input_cost + output_cost
    return round(total_cost, 4)


def make_openai_call(prompt: str, system_message: str, model: str = "gpt-4o-mini", 
                    max_tokens: int = 300, temperature: float = 0.3, return_usage: bool = False):
    """
    Make an OpenAI API call with error handling.
    Handles reasoning models (o1, o3, etc.) differently from standard models.
    """
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Check if this is a reasoning model
        if is_reasoning_model(model):
            # For reasoning models, use the responses API endpoint
            reasoning_effort = st.session_state.get('reasoning_effort_setting', 'low')
            text_verbosity = st.session_state.get('text_verbosity_setting', 'low')
            
            # Handle model-specific verbosity requirements
            if model in ['o3', 'o3-mini', 'o4-mini']:
                if text_verbosity == 'low':
                    text_verbosity = 'medium'
            
            # Adjust max_output_tokens based on reasoning effort
            if reasoning_effort == 'high':
                adjusted_max_tokens = max(max_tokens * 2, 4000)
                max_tokens = adjusted_max_tokens
            
            response = client.responses.create(
                model=model,
                input=prompt,
                instructions=system_message,
                max_output_tokens=max_tokens,
                reasoning={"effort": reasoning_effort},
                text={"verbosity": text_verbosity}
            )
            
            # Extract content from reasoning model response
            content = None
            if hasattr(response, 'output') and response.output:
                for output_item in response.output:
                    if getattr(output_item, 'type', '') != 'reasoning':
                        if hasattr(output_item, 'content') and output_item.content:
                            for content_item in output_item.content:
                                if hasattr(content_item, 'text'):
                                    content = content_item.text.strip()
                                    break
                        if content:
                            break
            
            if content is None:
                raise Exception("Could not extract content from reasoning model response")
            
            if return_usage:
                usage = response.usage
                usage_dict = {
                    'prompt_tokens': getattr(usage, 'input_tokens', 0),
                    'completion_tokens': getattr(usage, 'output_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0)
                }
                return content, usage_dict
            else:
                return content
        else:
            # Standard models use the chat completions endpoint
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = response.choices[0].message.content.strip()
            if return_usage:
                usage = response.usage
                usage_dict = {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens
                }
                return content, usage_dict
            else:
                return content
            
    except openai.BadRequestError as e:
        if "context_length_exceeded" in str(e).lower() or "maximum context length" in str(e).lower():
            raise ValueError("Context length exceeded")
        else:
            raise Exception(f"API error: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")


def track_actual_cost(input_tokens: int, output_tokens: int, model: str) -> Dict[str, any]:
    """
    Track actual cost for analysis and return cost summary.
    """
    actual_cost = calculate_cost(input_tokens, output_tokens, model)
    
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
        'actual_cost': actual_cost,
        'model': model,
        'cost_per_1k_tokens': actual_cost / ((input_tokens + output_tokens) / 1000) if (input_tokens + output_tokens) > 0 else 0
    }
