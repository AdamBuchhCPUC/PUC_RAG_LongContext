"""
Processing Tab
Handles document processing functionality.
"""

import streamlit as st
from pathlib import Path


def create_processing_tab():
    """Create the document processing tab"""
    st.header("🔄 Process Documents")
    st.markdown("Convert downloaded PDFs to searchable text and create vector embeddings")
    
    # Check if documents exist
    documents_folder = "./documents"
    if not Path(documents_folder).exists():
        st.warning("No documents folder found. Please download some documents first.")
        return
    
    # Check if data is already loaded in session state
    vector_store_exists = st.session_state.get('vector_store') is not None
    documents_exists = st.session_state.get('documents') is not None
    metadata_exists = st.session_state.get('metadata') is not None and len(st.session_state.get('metadata', {})) > 0
    
    session_has_data = vector_store_exists and documents_exists and metadata_exists
    
    # Always load metadata from files to get actual document count
    from processing.document_downloader import load_documents_metadata, check_if_processing_needed
    metadata = load_documents_metadata(documents_folder)
    
    if not metadata:
        st.warning("No documents found in the documents folder.")
        return
    
    # Check processing status
    processing_status = check_if_processing_needed(documents_folder)
    
    # Show status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", processing_status['total_documents'])
    with col2:
        st.metric("Processed", processing_status['processed'])
    with col3:
        st.metric("Unprocessed", processing_status['unprocessed'])
    
    # Show current session state
    if session_has_data:
        st.success("✅ **Data is loaded in memory and ready for Q&A!**")
        st.info(f"📊 **Session Data**: {len(st.session_state.documents)} chunks from {len(st.session_state.metadata)} documents")
        
        if st.button("🔄 Reload Data from Disk"):
            try:
                from processing.document_processor import DocumentProcessor
                processor = DocumentProcessor()
                vector_store, bm25, documents, metadata = processor.load_processed_data("./processed_data")
                
                if vector_store is not None and bm25 is not None and documents and metadata:
                    st.session_state.vector_store = vector_store
                    st.session_state.bm25 = bm25
                    st.session_state.documents = documents
                    st.session_state.metadata = metadata
                    st.success("✅ Data reloaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Could not load data from disk")
            except Exception as e:
                st.error(f"❌ Error reloading data: {e}")
    else:
        st.warning("⚠️ **No data loaded in memory**")
        
        # Check if processed data exists on disk
        processed_data_exists = (
            Path("./processed_data_vector").exists() and
            Path("./processed_data_vector/processed_data_bm25.pkl").exists() and
            Path("./processed_data_vector/processed_data_documents.pkl").exists() and
            Path("./processed_data_vector/processed_data_metadata.json").exists()
        )
        
        if processed_data_exists:
            st.info("📁 **Processed data found on disk** - Click 'Load Processed Data' to use it")
            
            if st.button("📁 Load Processed Data", type="primary"):
                try:
                    from processing.document_processor import DocumentProcessor
                    processor = DocumentProcessor()
                    vector_store, bm25, documents, metadata = processor.load_processed_data("./processed_data")
                    
                    if vector_store is not None and bm25 is not None and documents and metadata:
                        st.session_state.vector_store = vector_store
                        st.session_state.bm25 = bm25
                        st.session_state.documents = documents
                        st.session_state.metadata = metadata
                        st.success("✅ Data loaded successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Could not load data from disk")
                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")
        else:
            st.info("📄 **No processed data found** - Process documents to create searchable text")
    
    # Processing section
    st.subheader("🔄 Process Documents")
    
    if processing_status['needs_processing']:
        st.info(f"📄 **{processing_status['unprocessed']} documents need processing**")
        
        # Get model from sidebar
        model = st.session_state.get('selected_model', 'gpt-4o-mini')
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Selected Model**: {model}")
        with col2:
            if st.button("🚀 Start Processing", type="primary"):
                try:
                    from processing.document_processor import DocumentProcessor
                    
                    # Initialize processor
                    processor = DocumentProcessor(model=model)
                    
                    # Process documents
                    with st.spinner("🔄 Processing documents..."):
                        vector_store, bm25, documents, metadata = processor.process_documents(documents_folder)
                    
                    if vector_store is not None and bm25 is not None and documents and metadata:
                        # Save processed data
                        processor.save_processed_data(vector_store, bm25, documents, metadata)
                        
                        # Update session state
                        st.session_state.vector_store = vector_store
                        st.session_state.bm25 = bm25
                        st.session_state.documents = documents
                        st.session_state.metadata = metadata
                        
                        st.success("✅ Processing complete! Data is now ready for Q&A.")
                        st.rerun()
                    else:
                        st.error("❌ Processing failed - no data was created")
                        
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
    else:
        st.success("✅ **All documents are already processed!**")
        st.info("💡 Use the 'Ask Questions' tab to start querying your documents.")
    
    # Cache management
    st.subheader("🗄️ Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Processed Data"):
            try:
                from processing.document_processor import DocumentProcessor
                processor = DocumentProcessor()
                if processor.clear_processed_data():
                    # Clear session state
                    st.session_state.vector_store = None
                    st.session_state.bm25 = None
                    st.session_state.documents = None
                    st.session_state.metadata = {}
                    st.success("✅ All processed data cleared!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing data: {e}")
    
    with col2:
        if st.button("📊 Show Cache Stats"):
            try:
                from processing.cache_manager import ContentHashCache
                cache = ContentHashCache()
                stats = cache.get_cache_stats()
                st.info(f"📊 **Cache Statistics**: {stats['cached_embeddings']} cached embeddings ({stats['cache_size_mb']:.2f} MB)")
            except Exception as e:
                st.error(f"❌ Error getting cache stats: {e}")
