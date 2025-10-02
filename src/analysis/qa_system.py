"""
Simplified Q&A system for PUC RAG (LC) System.
Handles question answering using both vector and keyword search.
"""

import streamlit as st
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.schema import Document
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from analysis.llm_utils import calculate_cost, make_openai_call


def ask_question(
    question: str,
    vector_store,
    bm25,
    documents: List[Document],
    metadata: Dict[str, Any],
    model: str,
    search_type: str = "Hybrid (Recommended)",
    num_results: int = 10
) -> Tuple[str, List[Dict[str, Any]]]:
    """Ask a question and get an answer with sources"""
    
    if not question.strip():
        return "Please enter a question.", []
    
    st.write(f"🤖 **Model**: {model}")
    st.write(f"📚 **Available documents**: {len(documents)}")
    
    # Get relevant documents based on search type
    if search_type == "Vector Search":
        relevant_docs = _vector_search(question, vector_store, num_results)
    elif search_type == "Keyword Search":
        relevant_docs = _keyword_search(question, bm25, documents, num_results)
    else:  # Hybrid
        relevant_docs = _hybrid_search(question, vector_store, bm25, documents, num_results)
    
    if not relevant_docs:
        return "No relevant documents found for your question.", []
    
    # Prepare context for LLM
    context = _prepare_context(relevant_docs)
    
    # Generate answer using LLM
    answer = _generate_answer(question, context, model)
    
    # Prepare sources
    sources = _prepare_sources(relevant_docs, metadata)
    
    return answer, sources


def _vector_search(question: str, vector_store, num_results: int) -> List[Document]:
    """Perform vector similarity search"""
    try:
        docs = vector_store.similarity_search(question, k=num_results)
        return docs
    except Exception as e:
        st.error(f"Vector search error: {e}")
        return []


def _keyword_search(question: str, bm25, documents: List[Document], num_results: int) -> List[Document]:
    """Perform keyword search using BM25"""
    try:
        # Tokenize question
        question_tokens = word_tokenize(question.lower())
        
        # Get BM25 scores
        scores = bm25.get_scores(question_tokens)
        
        # Get top documents
        top_indices = np.argsort(scores)[::-1][:num_results]
        
        relevant_docs = []
        for idx in top_indices:
            if idx < len(documents):
                relevant_docs.append(documents[idx])
        
        return relevant_docs
    except Exception as e:
        st.error(f"Keyword search error: {e}")
        return []


def _hybrid_search(question: str, vector_store, bm25, documents: List[Document], num_results: int) -> List[Document]:
    """Perform hybrid search combining vector and keyword search"""
    try:
        # Get vector results
        vector_docs = _vector_search(question, vector_store, num_results)
        
        # Get keyword results
        keyword_docs = _keyword_search(question, bm25, documents, num_results)
        
        # Combine and deduplicate
        all_docs = vector_docs + keyword_docs
        seen_content = set()
        unique_docs = []
        
        for doc in all_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
                if len(unique_docs) >= num_results:
                    break
        
        return unique_docs
    except Exception as e:
        st.error(f"Hybrid search error: {e}")
        return []


def _prepare_context(documents: List[Document]) -> str:
    """Prepare context from relevant documents"""
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        # Get source information
        source = doc.metadata.get('source', 'Unknown')
        proceeding = doc.metadata.get('proceeding', 'Unknown')
        doc_type = doc.metadata.get('document_type', 'Unknown')
        
        # Add document header
        context_parts.append(f"--- Document {i} ---")
        context_parts.append(f"Source: {source}")
        context_parts.append(f"Proceeding: {proceeding}")
        context_parts.append(f"Type: {doc_type}")
        context_parts.append("")
        context_parts.append(doc.page_content)
        context_parts.append("")
    
    return "\n".join(context_parts)


def _generate_answer(question: str, context: str, model: str) -> str:
    """Generate answer using LLM"""
    try:
        system_message = """You are an expert CPUC regulatory analyst. Answer questions based on the provided document context. 

Guidelines:
- Provide accurate, well-sourced answers based on the documents
- Cite specific documents and proceedings when relevant
- If the answer is not in the provided context, say so clearly
- Be concise but comprehensive
- Focus on factual information from the documents"""

        prompt = f"""Based on the following CPUC documents, please answer this question: {question}

DOCUMENTS:
{context}

Please provide a clear, well-sourced answer based on the document content."""

        # Make LLM call
        answer = make_openai_call(
            prompt=prompt,
            system_message=system_message,
            model=model,
            max_tokens=1000,
            temperature=0.3
        )
        
        return answer
        
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while generating the answer."


def _prepare_sources(documents: List[Document], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare source information for display"""
    sources = []
    
    for i, doc in enumerate(documents, 1):
        source_info = {
            'rank': i,
            'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            'source': doc.metadata.get('source', 'Unknown'),
            'proceeding': doc.metadata.get('proceeding', 'Unknown'),
            'document_type': doc.metadata.get('document_type', 'Unknown'),
            'filed_by': doc.metadata.get('filed_by', 'Unknown'),
            'chunk_level': doc.metadata.get('chunk_level', 'Unknown'),
            'page_numbers': doc.metadata.get('page_numbers', [])
        }
        sources.append(source_info)
    
    return sources
