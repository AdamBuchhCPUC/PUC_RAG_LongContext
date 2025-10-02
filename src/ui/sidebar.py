"""
Sidebar configuration for PUC RAG (LC) System.
Handles model selection and basic configuration.
"""

import streamlit as st
from typing import Dict, Any


def create_sidebar() -> Dict[str, Any]:
    """Create and configure the sidebar with all options"""
    
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection with pricing and TPM info
    st.sidebar.subheader("🤖 AI Model")
    
    # Import model configuration
    from analysis.model_config import get_model_limits
    from analysis.llm_utils import calculate_cost
    
    # Model options with detailed pricing and TPM info
    model_options = {
        # Most Advanced Models (Top Tier)
        "o3 - $2.00/$8.00 (30K TPM) ⭐": "o3",
        "o3-mini - $1.10/$4.40 (200K TPM) ⭐": "o3-mini",
        "gpt-5 - $1.25/$10.00 (500K TPM) ⭐": "gpt-5",
        "gpt-5-mini - $0.25/$2.00 (500K TPM) ⭐": "gpt-5-mini",
        
        # Advanced Reasoning Models
        "o1 - $15.00/$60.00 (30K TPM)": "o1",
        "o1-mini - $1.10/$4.40 (200K TPM)": "o1-mini",
        "o1-pro - $15.00/$60.00 (30K TPM)": "o1-pro",
        
        # GPT-4o Models (Recommended)
        "gpt-4o - $2.50/$10.00 (30K TPM) 🚀": "gpt-4o",
        "gpt-4o-mini - $0.15/$0.60 (200K TPM) 💰": "gpt-4o-mini",
        
        # GPT-4.1 Models
        "gpt-4.1 - $2.00/$8.00 (30K TPM)": "gpt-4.1",
        "gpt-4.1-mini - $0.40/$1.60 (200K TPM)": "gpt-4.1-mini",
        
        # GPT-5 Models
        "gpt-5-nano - $0.05/$0.40 (200K TPM)": "gpt-5-nano",
        
        # Legacy Models
        "gpt-4 - $10.00/$30.00 (10K TPM)": "gpt-4",
        "gpt-3.5-turbo - $1.00/$2.00 (200K TPM)": "gpt-3.5-turbo"
    }
    
    # Get current model from session state
    current_model = st.session_state.get('selected_model', 'gpt-4o-mini')
    
    # Model selection
    selected_model_display = st.sidebar.selectbox(
        "Select AI Model:",
        options=list(model_options.keys()),
        index=list(model_options.values()).index(current_model) if current_model in model_options.values() else 0,
        help="Choose the AI model for processing and Q&A. Pricing shown as input/output per 1K tokens."
    )
    
    selected_model = model_options[selected_model_display]
    st.session_state.selected_model = selected_model
    
    # Show model info
    limits = get_model_limits(selected_model)
    st.sidebar.info(f"**Selected Model**: {selected_model}")
    st.sidebar.info(f"**TPM Limit**: {limits.tpm:,}")
    st.sidebar.info(f"**RPM Limit**: {limits.rpm:,}")
    
    # Reasoning model settings
    if limits.is_reasoning_model:
        st.sidebar.subheader("🧠 Reasoning Settings")
        
        reasoning_effort = st.sidebar.selectbox(
            "Reasoning Effort:",
            options=["low", "medium", "high"],
            index=0,
            help="Higher effort = more thorough reasoning but slower and more expensive"
        )
        st.session_state.reasoning_effort_setting = reasoning_effort
        
        text_verbosity = st.sidebar.selectbox(
            "Text Verbosity:",
            options=["low", "medium", "high"],
            index=0,
            help="How verbose the model's responses should be"
        )
        st.session_state.text_verbosity_setting = text_verbosity
    
    # Smart Query Agent configuration
    st.sidebar.subheader("🧠 Smart Query Agent")
    
    st.sidebar.info("🤖 **Smart Agent Active**")
    st.sidebar.info("• **Classification**: GPT-4o-mini")
    st.sidebar.info("• **Summary Queries**: Multi-stage processing")
    st.sidebar.info("• **Other Queries**: Vector/BM25 search")
    st.sidebar.info("• **Context Management**: Dynamic based on model and query type")
    
    # Cost tracking
    st.sidebar.subheader("💰 Cost Tracking")
    
    if st.session_state.get('total_qa_costs', 0) > 0:
        st.sidebar.metric(
            "Total Q&A Costs",
            f"${st.session_state.total_qa_costs:.4f}"
        )
    
    if st.sidebar.button("🔄 Reset Costs"):
        st.session_state.total_qa_costs = 0
        st.rerun()
    
    # Cache management
    st.sidebar.subheader("🗄️ Cache Management")
    
    if st.sidebar.button("🗑️ Clear Processed Data"):
        try:
            from processing.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            if processor.clear_processed_data():
                st.session_state.vector_store = None
                st.session_state.bm25 = None
                st.session_state.documents = None
                st.session_state.metadata = {}
                st.rerun()
        except Exception as e:
            st.error(f"Error clearing data: {e}")
    
    # Return configuration
    return {
        'model': selected_model,
        'reasoning_effort': st.session_state.get('reasoning_effort_setting', 'low'),
        'text_verbosity': st.session_state.get('text_verbosity_setting', 'low')
    }
