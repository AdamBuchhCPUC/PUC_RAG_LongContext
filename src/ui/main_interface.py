"""
Main Streamlit interface for PUC RAG (LC) System.
Streamlined version with 3-tab interface for deployment.
"""

import streamlit as st
import os
from pathlib import Path
from typing import Dict, Any

# Import tab components
from ui.tabs.download_tab import create_download_tab
from ui.tabs.processing_tab import create_processing_tab
from ui.tabs.qa_tab import create_qa_tab
from ui.tabs.persistence_tab import create_persistence_tab
from ui.sidebar import create_sidebar


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'bm25' not in st.session_state:
        st.session_state.bm25 = None
    if 'documents' not in st.session_state:
        st.session_state.documents = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = {}
    if 'total_qa_costs' not in st.session_state:
        st.session_state.total_qa_costs = 0


def check_api_key():
    """Check if OpenAI API key is available"""
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found in environment variables!")
        st.info("Please make sure your .env file contains: OPENAI_API_KEY=your_key_here")
        return False
    return True


def auto_load_processed_data():
    """Auto-load processed data if available"""
    # Check each required file
    vector_dir_exists = Path("./processed_data_vector").exists() and len(list(Path("./processed_data_vector").rglob("*"))) > 0
    bm25_file_exists = Path("./processed_data_vector/processed_data_bm25.pkl").exists()
    documents_file_exists = Path("./processed_data_vector/processed_data_documents.pkl").exists()
    metadata_file_exists = Path("./processed_data_vector/processed_data_metadata.json").exists()

    if (st.session_state.vector_store is None and
        vector_dir_exists and
        bm25_file_exists and
        documents_file_exists and
        metadata_file_exists):

        try:
            # Import here to avoid circular imports
            from processing.document_processor import DocumentProcessor

            processor = DocumentProcessor()
            vector_store, bm25, documents, metadata = processor.load_processed_data("./processed_data")

            if vector_store is not None and bm25 is not None and documents and metadata:
                # Update session state
                st.session_state.vector_store = vector_store
                st.session_state.bm25 = bm25
                st.session_state.documents = documents
                st.session_state.metadata = metadata

                st.success(f"✅ Auto-loaded {len(documents)} document chunks from previous session!")

        except Exception as e:
            st.error(f"❌ Could not auto-load processed data: {e}")


def main():
    """Main application function"""
    st.set_page_config(
        page_title="PUC RAG (LC) - CPUC Document Q&A System",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ PUC RAG (LC) - CPUC Document Q&A System")
    st.markdown("Ask questions about CPUC decisions and proceedings with intelligent search and caching")
    
    # Check API key
    if not check_api_key():
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Auto-load processed data
    auto_load_processed_data()
    
    # Create sidebar configuration
    st.session_state.sidebar_config = create_sidebar()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📥 Download Documents", 
        "🔄 Process Documents", 
        "🔍 Ask Questions",
        "💾 Data Persistence"
    ])
    
    with tab1:
        create_download_tab()
    
    with tab2:
        create_processing_tab()
    
    with tab3:
        create_qa_tab()
    
    with tab4:
        create_persistence_tab()


if __name__ == "__main__":
    main()
