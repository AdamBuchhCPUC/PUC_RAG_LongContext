"""
Model configuration for TPM limits and chunking strategies.
Compatible with future batch queuing and flex processing.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ModelLimits:
    """Model rate limits and capabilities."""
    tpm: int  # Tokens per minute
    rpm: int  # Requests per minute
    rpd: int  # Requests per day
    tpd: int  # Tokens per day
    context_window: int  # Maximum context window in tokens
    supports_batch: bool = False
    supports_flex: bool = False
    is_reasoning_model: bool = False  # Whether this is a reasoning model (o1, o3, etc.)

# Model configurations with TPM limits
MODEL_LIMITS: Dict[str, ModelLimits] = {
    # GPT-3.5 Turbo models
    "gpt-3.5-turbo": ModelLimits(2000000, 5000, 0, 20000000, 16384, supports_batch=True),
    "gpt-3.5-turbo-0125": ModelLimits(2000000, 5000, 0, 20000000, 16384, supports_batch=True),
    "gpt-3.5-turbo-1106": ModelLimits(2000000, 5000, 0, 20000000, 16384, supports_batch=True),
    "gpt-3.5-turbo-16k": ModelLimits(2000000, 5000, 0, 20000000, 16384, supports_batch=True),
    "gpt-3.5-turbo-instruct": ModelLimits(90000, 3500, 0, 200000, 4096, supports_batch=True),
    "gpt-3.5-turbo-instruct-0914": ModelLimits(90000, 3500, 0, 200000, 4096, supports_batch=True),
    
    # GPT-4 models
    "gpt-4": ModelLimits(40000, 5000, 0, 200000, 8192),
    "gpt-4-0613": ModelLimits(40000, 5000, 0, 200000, 8192),
    "gpt-4-turbo": ModelLimits(450000, 500, 0, 1350000, 128000),
    "gpt-4-turbo-2024-04-09": ModelLimits(450000, 500, 0, 1350000, 128000),
    "gpt-4-turbo-preview": ModelLimits(450000, 500, 0, 1350000, 128000),
    "gpt-4-0125-preview": ModelLimits(450000, 500, 0, 1350000, 128000),
    "gpt-4-1106-preview": ModelLimits(450000, 500, 0, 1350000, 128000),
    
    # GPT-4.1 models
    "gpt-4.1-2025-04-14": ModelLimits(450000, 5000, 0, 1350000, 128000),
    "gpt-4.1": ModelLimits(450000, 5000, 0, 1350000, 128000),
    "gpt-4.1-long-context": ModelLimits(500000, 250, 0, 20000000, 1000000, supports_flex=True),
    "gpt-4.1-mini": ModelLimits(2000000, 5000, 0, 20000000, 128000, supports_batch=True),
    "gpt-4.1-mini-2025-04-14": ModelLimits(2000000, 5000, 0, 20000000, 128000, supports_batch=True),
    "gpt-4.1-mini-long-context": ModelLimits(1000000, 500, 0, 40000000, 1000000, supports_flex=True),
    "gpt-4.1-nano": ModelLimits(2000000, 5000, 0, 20000000, 128000, supports_batch=True),
    "gpt-4.1-nano-2025-04-14": ModelLimits(2000000, 5000, 0, 20000000, 128000, supports_batch=True),
    "gpt-4.1-nano-long-context": ModelLimits(1000000, 500, 0, 40000000, 1000000, supports_flex=True),
    
    # GPT-4o models
    "gpt-4o": ModelLimits(450000, 5000, 0, 1350000, 128000),
    "gpt-4o-2024-05-13": ModelLimits(450000, 5000, 0, 1350000, 128000),
    "gpt-4o-2024-08-06": ModelLimits(450000, 5000, 0, 1350000, 128000),
    "gpt-4o-2024-11-20": ModelLimits(450000, 5000, 0, 1350000, 128000),
    "gpt-4o-audio-preview": ModelLimits(450000, 5000, 0, 1350000, 128000),
    "gpt-4o-audio-preview-2024-10-01": ModelLimits(450000, 5000, 0, 1350000, 128000),
    "gpt-4o-audio-preview-2024-12-17": ModelLimits(450000, 5000, 0, 1350000, 128000),
    "gpt-4o-audio-preview-2025-06-03": ModelLimits(250000, 3000, 0, 0, 128000),
    "gpt-4o-mini": ModelLimits(2000000, 5000, 0, 20000000, 128000, supports_batch=True),
    "gpt-4o-mini-2024-07-18": ModelLimits(2000000, 5000, 0, 20000000, 128000, supports_batch=True),
    "gpt-4o-mini-audio-preview": ModelLimits(2000000, 5000, 0, 20000000, 128000, supports_batch=True),
    "gpt-4o-mini-audio-preview-2024-12-17": ModelLimits(2000000, 5000, 0, 20000000, 128000, supports_batch=True),
    "gpt-4o-mini-search-preview": ModelLimits(45000, 500, 0, 0, 128000),
    "gpt-4o-mini-search-preview-2025-03-11": ModelLimits(45000, 500, 0, 0, 128000),
    "gpt-4o-mini-transcribe": ModelLimits(150000, 2000, 0, 0, 128000),
    "gpt-4o-search-preview": ModelLimits(45000, 500, 0, 0, 128000),
    "gpt-4o-search-preview-2025-03-11": ModelLimits(45000, 500, 0, 0, 128000),
    "gpt-4o-transcribe": ModelLimits(100000, 2000, 0, 0, 128000),
    
    # GPT-5 models
    "gpt-5": ModelLimits(1000000, 5000, 0, 3000000, 256000, supports_flex=True, is_reasoning_model=True),
    "gpt-5-2025-08-07": ModelLimits(1000000, 5000, 0, 3000000, 256000, supports_flex=True, is_reasoning_model=True),
    "gpt-5-chat-latest": ModelLimits(450000, 5000, 0, 1350000, 256000, is_reasoning_model=True),
    "gpt-5-codex": ModelLimits(1000000, 5000, 0, 1350000, 256000, supports_flex=True, is_reasoning_model=True),
    "gpt-5-mini": ModelLimits(2000000, 5000, 0, 20000000, 256000, supports_batch=True, supports_flex=True, is_reasoning_model=True),
    "gpt-5-mini-2025-08-07": ModelLimits(2000000, 5000, 0, 20000000, 256000, supports_batch=True, supports_flex=True, is_reasoning_model=True),
    "gpt-5-nano": ModelLimits(2000000, 5000, 0, 20000000, 256000, supports_batch=True, is_reasoning_model=True),
    "gpt-5-nano-2025-08-07": ModelLimits(2000000, 5000, 0, 20000000, 256000, supports_batch=True, is_reasoning_model=True),
    "gpt-5-pro": ModelLimits(450000, 5000, 0, 1350000, 256000, is_reasoning_model=True),
    "gpt-5-pro-2025-10-06": ModelLimits(450000, 5000, 0, 1350000, 256000, is_reasoning_model=True),
    
    # Other models
    "gpt-audio": ModelLimits(250000, 3000, 0, 0, 128000),
    "gpt-audio-2025-08-28": ModelLimits(250000, 3000, 0, 0, 128000),
    "gpt-audio-mini": ModelLimits(250000, 3000, 0, 0, 128000),
    "gpt-audio-mini-2025-10-06": ModelLimits(250000, 3000, 0, 0, 128000),
    "chatgpt-4o-latest": ModelLimits(500000, 200, 0, 0, 128000),
    "o1": ModelLimits(30000, 500, 90000, 0, 200000, is_reasoning_model=True),
    "o1-2024-12-17": ModelLimits(30000, 500, 90000, 0, 200000, is_reasoning_model=True),
    "o1-mini": ModelLimits(200000, 500, 2000000, 0, 200000, supports_batch=True, is_reasoning_model=True),
    "o1-mini-2024-09-12": ModelLimits(200000, 500, 2000000, 0, 200000, supports_batch=True, is_reasoning_model=True),
    "o1-pro": ModelLimits(30000, 500, 90000, 0, 200000, is_reasoning_model=True),
    "o1-pro-2025-03-19": ModelLimits(30000, 500, 90000, 0, 200000, is_reasoning_model=True),
    "o3": ModelLimits(30000, 500, 90000, 0, 200000, is_reasoning_model=True),
    "o3-2025-04-16": ModelLimits(30000, 500, 90000, 0, 200000, is_reasoning_model=True),
    "o3-mini": ModelLimits(200000, 500, 2000000, 0, 200000, supports_batch=True, is_reasoning_model=True),
    "o3-mini-2025-01-31": ModelLimits(200000, 500, 2000000, 0, 200000, supports_batch=True, is_reasoning_model=True),
    "o4-mini": ModelLimits(200000, 500, 2000000, 0, 200000, supports_batch=True, is_reasoning_model=True),
    "o4-mini-2025-04-16": ModelLimits(200000, 500, 2000000, 0, 200000, supports_batch=True, is_reasoning_model=True),
    "o4-mini-deep-research": ModelLimits(200000, 500, 200000, 0, 200000, is_reasoning_model=True),
    "o4-mini-deep-research-2025-06-26": ModelLimits(250000, 3000, 0, 0, 200000, is_reasoning_model=True),
}

def get_model_limits(model_name: str) -> ModelLimits:
    """Get rate limits for a specific model."""
    # Handle model name variations and aliases
    model_name = model_name.lower().strip()
    
    # Direct lookup
    if model_name in MODEL_LIMITS:
        return MODEL_LIMITS[model_name]
    
    # Handle common aliases and variations
    aliases = {
        "gpt-4o": "gpt-4o",
        "gpt-4": "gpt-4",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-4.1": "gpt-4.1-2025-04-14",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4.1-nano": "gpt-4.1-nano",
        "gpt-5": "gpt-5",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5-nano": "gpt-5-nano",
    }
    
    if model_name in aliases:
        return MODEL_LIMITS[aliases[model_name]]
    
    # Default to conservative limits if model not found
    return ModelLimits(10000, 500, 10000, 100000)

def get_chunking_strategy(model_name: str) -> Tuple[int, int]:
    """
    Get chunking strategy based on model TPM limits.
    
    Returns:
        Tuple of (single_analysis_threshold_tokens, max_section_tokens)
        - single_analysis_threshold_tokens: Max tokens for single analysis
        - max_section_tokens: Max tokens per section when splitting
    """
    limits = get_model_limits(model_name)
    
    # Calculate aggressive thresholds based on actual TPM limits
    # Use 90% of TPM limit for maximum efficiency
    safe_tpm = int(limits.tpm * 0.9)
    
    # Set thresholds based on BOTH TPM limits AND context windows
    if limits.tpm >= 500000:  # Ultra high TPM models (GPT-5, etc.)
        # For GPT-5 with 256K context window, we can be much more aggressive
        if "gpt-5" in model_name.lower():
            # GPT-5: Context window is the limiting factor, not TPM
            context_limit = int(limits.context_window * 0.8)  # Use actual context window
            tpm_limit = safe_tpm // 2  # TPM-based limit
            single_threshold_tokens = min(context_limit, tpm_limit)  # Use more restrictive
            max_section_tokens = min(100000, single_threshold_tokens // 2)
        else:
            # Other ultra-high TPM models: TPM might still be limiting
            single_threshold_tokens = min(50000, safe_tpm // 2)   # 50k tokens max
            max_section_tokens = min(25000, safe_tpm // 4)        # 25k tokens max
    elif limits.tpm >= 200000:  # High TPM models (GPT-3.5, GPT-4.1-mini, etc.)
        # High TPM models: Context window might be limiting
        context_limit = int(limits.context_window * 0.8)  # Use actual context window
        tpm_limit = safe_tpm // 2  # TPM-based limit
        single_threshold_tokens = min(context_limit, tpm_limit)  # Use more restrictive
        max_section_tokens = min(50000, single_threshold_tokens // 2)
    elif limits.tpm >= 100000:  # Medium TPM models
        # Medium TPM models: TPM is likely limiting
        single_threshold_tokens = min(25000, safe_tpm // 2)  # 25k tokens max
        max_section_tokens = min(12500, safe_tpm // 4)       # 12.5k tokens max
    elif limits.tpm >= 30000:   # Standard TPM models (GPT-4o, GPT-4.1)
        # Special case for GPT-4o: be very aggressive since it's high quality
        if "gpt-4o" in model_name.lower():
            # For GPT-4o, use a much higher threshold since it's high quality
            single_threshold_tokens = 25000  # 25k tokens max for GPT-4o (ignore TPM limit)
            max_section_tokens = 12500      # 12.5k tokens max for GPT-4o
        else:
            single_threshold_tokens = min(12500, safe_tpm // 2)  # 12.5k tokens max
            max_section_tokens = min(6250, safe_tpm // 4)        # 6.25k tokens max
    else:  # Low TPM models (GPT-4, older models)
        single_threshold_tokens = min(5000, safe_tpm // 2)   # 5k tokens max
        max_section_tokens = min(2500, safe_tpm // 4)        # 2.5k tokens max
    
    return single_threshold_tokens, max_section_tokens

def supports_batch_processing(model_name: str) -> bool:
    """Check if model supports batch processing."""
    return get_model_limits(model_name).supports_batch

def supports_flex_processing(model_name: str) -> bool:
    """Check if model supports flex processing."""
    return get_model_limits(model_name).supports_flex

def is_reasoning_model(model_name: str) -> bool:
    """Check if model is a reasoning model (o1, o3, etc.)."""
    return get_model_limits(model_name).is_reasoning_model

def get_model_capabilities(model_name: str) -> Dict[str, bool]:
    """Get all capabilities for a model."""
    limits = get_model_limits(model_name)
    return {
        "batch_processing": limits.supports_batch,
        "flex_processing": limits.supports_flex,
        "reasoning_model": limits.is_reasoning_model,
        "high_tpm": limits.tpm >= 100000,
        "medium_tpm": 30000 <= limits.tpm < 100000,
        "low_tpm": limits.tpm < 30000,
    }

def get_optimal_chunk_size(model_name: str) -> int:
    """
    Get optimal chunk size based on model's context window and capabilities.
    Returns chunk size in characters (not tokens).
    """
    limits = get_model_limits(model_name)
    
    # For models with very large context windows, use much larger chunks
    if "gpt-5" in model_name.lower():
        # GPT-5 has 256K context window - use very large chunks
        return 50000  # 50k characters ≈ 12.5k tokens
    elif limits.tpm >= 200000:  # High TPM models
        return 20000  # 20k characters ≈ 5k tokens
    elif limits.tpm >= 100000:  # Medium TPM models
        return 10000  # 10k characters ≈ 2.5k tokens
    elif limits.tpm >= 30000:   # Standard models
        return 5000   # 5k characters ≈ 1.25k tokens
    else:  # Low TPM models
        return 2000   # 2k characters ≈ 500 tokens

def get_optimal_chunk_overlap(model_name: str) -> int:
    """
    Get optimal chunk overlap based on model's context window.
    Returns overlap in characters.
    """
    chunk_size = get_optimal_chunk_size(model_name)
    
    # Use 10% overlap for large context models, 20% for smaller ones
    if "gpt-5" in model_name.lower():
        return int(chunk_size * 0.1)  # 10% overlap for large chunks
    elif chunk_size >= 10000:
        return int(chunk_size * 0.15)  # 15% overlap for medium chunks
    else:
        return int(chunk_size * 0.2)  # 20% overlap for small chunks

def can_handle_large_documents(model_name: str) -> bool:
    """
    Check if model can handle very large documents in a single request.
    """
    if "gpt-5" in model_name.lower():
        return True  # GPT-5 has 256K context window
    elif get_optimal_chunk_size(model_name) >= 20000:
        return True  # High-TPM models can handle large chunks
    else:
        return False

def get_effective_context_limit(model_name: str) -> int:
    """
    Get the effective context limit considering both context window and TPM.
    Returns tokens that can actually be used in practice.
    """
    limits = get_model_limits(model_name)
    
    # Context window limits (use 80% of actual context window for safety)
    context_limit = int(limits.context_window * 0.8)
    
    # TPM limits (more restrictive for some models)
    tpm_limit = int(limits.tpm * 0.8)  # 80% of TPM for safety
    
    # Return the more restrictive limit
    return min(context_limit, tpm_limit)
