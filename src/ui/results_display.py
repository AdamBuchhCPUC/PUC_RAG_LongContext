"""
Results display functionality for PUC RAG (LC) System.
Handles display of search results and answers.
"""

import streamlit as st
from typing import List, Dict, Any


def display_results(answer: str, sources: List[Dict[str, Any]], question: str = ""):
    """Display Q&A results in a clean format"""
    
    if question:
        st.subheader(f"💬 Question: {question}")
    
    # Display answer
    st.subheader("💡 Answer")
    st.write(answer)
    
    # Display sources
    if sources:
        st.subheader("📚 Sources")
        
        for source in sources:
            with st.expander(f"Source {source['rank']}: {source['source']} ({source['document_type']})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Proceeding**: {source['proceeding']}")
                    st.write(f"**Filed by**: {source['filed_by']}")
                    st.write(f"**Chunk Level**: {source['chunk_level']}")
                    if source['page_numbers']:
                        st.write(f"**Pages**: {', '.join(map(str, source['page_numbers']))}")
                
                with col2:
                    st.write("**Content Preview**:")
                    st.text(source['content'])
    else:
        st.info("No sources found for this question.")


def display_processing_status(status: Dict[str, Any]):
    """Display document processing status"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", status.get('total_documents', 0))
    with col2:
        st.metric("Processed", status.get('processed', 0))
    with col3:
        st.metric("Unprocessed", status.get('unprocessed', 0))
    
    if status.get('needs_processing', False):
        st.warning("📄 Some documents need processing.")
    else:
        st.success("✅ All documents are processed and ready for Q&A!")


def display_cost_info(costs: Dict[str, Any]):
    """Display cost information"""
    
    if costs.get('total_cost', 0) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Cost", f"${costs['total_cost']:.4f}")
        with col2:
            st.metric("Input Tokens", f"{costs.get('input_tokens', 0):,}")
        with col3:
            st.metric("Output Tokens", f"{costs.get('output_tokens', 0):,}")
    else:
        st.info("💰 No costs tracked yet.")


def display_model_info(model: str, search_type: str, num_results: int):
    """Display current model and search configuration"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Model**: {model}")
    with col2:
        st.write(f"**Search**: {search_type}")
    with col3:
        st.write(f"**Results**: {num_results}")
