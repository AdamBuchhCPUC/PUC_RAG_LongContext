"""
Q&A Tab
Handles the question and answer interface for querying processed documents.
Uses Smart Query Agent for intelligent query processing.
"""

import streamlit as st
from analysis.smart_query_agent import SmartQueryAgent
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
    
    # Smart Query Agent info
    st.info("🧠 **Smart Query Agent Active**: Queries are intelligently classified and processed with multi-stage summarization for complex requests.")
    
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
                with st.spinner("🧠 Smart Agent processing query..."):
                    # Apply filters
                    filtered_documents = st.session_state.documents
                    filtered_metadata = st.session_state.metadata
                    
                    if selected_proceeding != "All Proceedings":
                        filtered_documents = [
                            doc for doc in filtered_documents 
                            if doc.metadata.get('proceeding') == selected_proceeding
                        ]
                        # Also filter metadata
                        filtered_metadata = {
                            k: v for k, v in filtered_metadata.items() 
                            if v.get('proceeding') == selected_proceeding
                        }
                    
                    if selected_doc_type != "All Types":
                        filtered_documents = [
                            doc for doc in filtered_documents 
                            if doc.metadata.get('document_type') == selected_doc_type
                        ]
                        # Also filter metadata
                        filtered_metadata = {
                            k: v for k, v in filtered_metadata.items() 
                            if v.get('document_type') == selected_doc_type
                        }
                    
                    # Initialize Smart Query Agent
                    smart_agent = SmartQueryAgent(filtered_documents, filtered_metadata)
                    
                    # Process query with Smart Agent
                    result = smart_agent.process_query(
                        question=question,
                        proceeding=selected_proceeding if selected_proceeding != "All Proceedings" else "",
                        model=model
                    )
                
                # Display answer
                st.subheader("💡 Answer")
                st.write(result['answer'])
                
                # Display processing stages if available
                if 'processing_stages' in result and result['processing_stages']:
                    st.subheader("🔄 Processing Stages")
                    for stage in result['processing_stages']:
                        with st.expander(f"Stage {stage.get('stage', 'N/A')}: {stage.get('description', 'Unknown')}"):
                            if 'documents_processed' in stage:
                                st.write(f"**Documents Processed**: {stage['documents_processed']}")
                            if 'cost' in stage:
                                st.write(f"**Cost**: ${stage['cost']:.4f}")
                            if 'model' in stage:
                                st.write(f"**Model**: {stage['model']}")
                            if 'output' in stage:
                                st.write("**Output**:")
                                if isinstance(stage['output'], list):
                                    for i, item in enumerate(stage['output'], 1):
                                        st.write(f"**{i}.** {item}")
                                else:
                                    st.write(stage['output'])
                
                # Display sources
                if result.get('sources'):
                    st.subheader("📚 Sources")
                    
                    for source in result['sources']:
                        with st.expander(f"Source: {source.get('source', 'Unknown')} ({source.get('document_type', 'Unknown')})"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write(f"**Type**: {source.get('type', 'Unknown')}")
                                st.write(f"**Proceeding**: {source.get('proceeding', 'Unknown')}")
                                st.write(f"**Filed by**: {source.get('filed_by', 'Unknown')}")
                                st.write(f"**Filing Date**: {source.get('filing_date', 'Unknown')}")
                                if source.get('response_to'):
                                    st.write(f"**Response to**: {source['response_to']}")
                            
                            with col2:
                                st.write("**Source Details**:")
                                st.text(f"Document: {source.get('source', 'Unknown')}")
                
                # Display cost breakdown
                if 'cost_breakdown' in result:
                    cost_breakdown = result['cost_breakdown']
                    st.subheader("💰 Cost Breakdown")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Cost", f"${cost_breakdown.get('total_cost', 0):.4f}")
                    with col2:
                        st.metric("API Calls", cost_breakdown.get('call_count', 0))
                    with col3:
                        st.metric("Model Used", model)
                    
                    # Update session state costs
                    st.session_state.total_qa_costs += cost_breakdown.get('total_cost', 0)
                
            except Exception as e:
                st.error(f"❌ Error processing question: {e}")
        else:
            st.warning("Please enter a question.")
    
    # Example questions
    st.subheader("💡 Example Questions")
    
    example_questions = [
        "Summarize this proceeding and all party responses",
        "What are the main issues discussed in this proceeding?",
        "What positions do the different parties take?",
        "What are the key findings and conclusions?",
        "What regulatory requirements are mentioned?",
        "What are the cost implications discussed?",
        "What environmental considerations are raised?",
        "What safety requirements are discussed?",
        "What are the proposed solutions or recommendations?",
        "Compare the different party positions on the main issues"
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
    
    # Smart Query Agent configuration
    st.subheader("⚙️ Smart Query Agent Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Model**: {model}")
    with col2:
        st.write(f"**Classification**: GPT-4o-mini")
    with col3:
        st.write(f"**Processing**: Intelligent routing")
    
    st.info("💡 **Tip**: The Smart Query Agent will automatically classify your question and route it to the appropriate processing method. For summary requests, it will use multi-stage analysis with document chains. For factual questions, it will use targeted search.")
