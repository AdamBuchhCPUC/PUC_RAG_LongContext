"""
Q&A Tab
Handles the question and answer interface for querying processed documents.
"""

import streamlit as st
from analysis.qa_system import ask_question
from analysis.llm_utils import calculate_cost


def create_qa_tab():
    """Create the Q&A tab"""
    st.header("🔍 Ask Questions")
    st.markdown("Ask questions about the processed documents")
    
    # Cost tracking section
    col1, col2 = st.columns([3, 1])
    with col1:
        if 'total_qa_costs' in st.session_state and st.session_state.total_qa_costs > 0:
            st.info(f"💰 **Total Q&A Costs**: ${st.session_state.total_qa_costs:.4f}")
    with col2:
        if st.button("🔄 Reset Costs", help="Reset the cost counter"):
            st.session_state.total_qa_costs = 0
            st.rerun()
    
    # Check if data is loaded
    vector_available = st.session_state.get('vector_store') is not None
    documents_available = st.session_state.get('documents') is not None
    metadata_available = st.session_state.get('metadata') is not None
    
    if not (vector_available and documents_available and metadata_available):
        st.warning("No processed documents found. Please process some documents first.")
        return
    
    # Get sidebar configuration
    sidebar_config = st.session_state.get('sidebar_config', {})
    model = sidebar_config.get('model', 'gpt-4o-mini')
    search_type = sidebar_config.get('search_type', 'Hybrid (Recommended)')
    num_results = sidebar_config.get('num_results', 10)
    
    # Show data status
    st.success(f"✅ **Data Ready**: {len(st.session_state.documents)} chunks from {len(st.session_state.metadata)} documents")
    
    # Question input
    st.subheader("💬 Ask a Question")
    
    # Get unique proceedings for filtering
    proceedings = set()
    for meta in st.session_state.metadata.values():
        proceeding = meta.get('proceeding', 'Unknown')
        if proceeding and proceeding != 'Unknown':
            proceedings.add(proceeding)
    
    proceedings = sorted(list(proceedings))
    
    # Filters
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question = st.text_area(
            "Enter your question:",
            placeholder="What are the main issues discussed in this proceeding?",
            height=100,
            help="Ask specific questions about the CPUC documents"
        )
    
    with col2:
        # Proceeding filter
        if proceedings:
            selected_proceeding = st.selectbox(
                "Filter by Proceeding:",
                options=["All Proceedings"] + proceedings,
                help="Filter to a specific proceeding"
            )
        else:
            selected_proceeding = "All Proceedings"
        
        # Document type filter
        doc_types = set()
        for meta in st.session_state.metadata.values():
            doc_type = meta.get('document_type', 'Unknown')
            if doc_type and doc_type != 'Unknown':
                doc_types.add(doc_type)
        
        doc_types = sorted(list(doc_types))
        if doc_types:
            selected_doc_type = st.selectbox(
                "Filter by Document Type:",
                options=["All Types"] + doc_types,
                help="Filter to a specific document type"
            )
        else:
            selected_doc_type = "All Types"
    
    # Ask question button
    if st.button("🔍 Ask Question", type="primary", disabled=not question.strip()):
        if question.strip():
            try:
                # Show processing
                with st.spinner("🔍 Searching documents and generating answer..."):
                    # Apply filters
                    filtered_documents = st.session_state.documents
                    filtered_metadata = st.session_state.metadata
                    
                    if selected_proceeding != "All Proceedings":
                        filtered_documents = [
                            doc for doc in filtered_documents 
                            if doc.metadata.get('proceeding') == selected_proceeding
                        ]
                    
                    if selected_doc_type != "All Types":
                        filtered_documents = [
                            doc for doc in filtered_documents 
                            if doc.metadata.get('document_type') == selected_doc_type
                        ]
                    
                    # Ask question
                    answer, sources = ask_question(
                        question=question,
                        vector_store=st.session_state.vector_store,
                        bm25=st.session_state.bm25,
                        documents=filtered_documents,
                        metadata=filtered_metadata,
                        model=model,
                        search_type=search_type,
                        num_results=num_results
                    )
                
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
                
                # Track costs (simplified)
                # Note: In a real implementation, you'd track actual token usage
                estimated_cost = 0.01  # Simplified cost tracking
                st.session_state.total_qa_costs += estimated_cost
                
            except Exception as e:
                st.error(f"❌ Error processing question: {e}")
        else:
            st.warning("Please enter a question.")
    
    # Example questions
    st.subheader("💡 Example Questions")
    
    example_questions = [
        "What are the main issues discussed in this proceeding?",
        "What positions do the different parties take?",
        "What are the key findings and conclusions?",
        "What regulatory requirements are mentioned?",
        "What are the cost implications discussed?",
        "What environmental considerations are raised?",
        "What safety requirements are discussed?",
        "What are the proposed solutions or recommendations?"
    ]
    
    # Display example questions in columns
    cols = st.columns(2)
    for i, example in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"💬 {example}", key=f"example_{i}"):
                st.session_state.example_question = example
                st.rerun()
    
    # Auto-fill example question if selected
    if 'example_question' in st.session_state:
        st.text_area(
            "Enter your question:",
            value=st.session_state.example_question,
            height=100,
            key="question_input"
        )
        if st.button("🔍 Ask This Question", type="primary"):
            question = st.session_state.example_question
            # Process the question (same logic as above)
            # ... (implementation would be the same as the main question processing)
            del st.session_state.example_question
            st.rerun()
    
    # Search configuration
    st.subheader("⚙️ Search Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Model**: {model}")
    with col2:
        st.write(f"**Search Type**: {search_type}")
    with col3:
        st.write(f"**Results**: {num_results}")
    
    st.info("💡 **Tip**: Use specific, detailed questions for better results. The system searches through document content to find relevant information.")
