"""
Smart Query Agent for CPUC Document Q&A System.
Handles query classification, document chain discovery, and multi-stage summarization.
"""

import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
from .document_relationships import DocumentRelationshipAnalyzer
from .llm_utils import make_openai_call, calculate_cost
import json
from datetime import datetime


class SmartQueryAgent:
    """Main orchestrator for intelligent query processing"""
    
    def __init__(self, documents: List[Document], metadata: Dict[str, Any]):
        """Initialize smart query agent"""
        self.documents = documents
        self.metadata = metadata
        self.relationship_analyzer = DocumentRelationshipAnalyzer(documents, metadata)
        
        # Cost tracking
        self.cost_tracker = {
            'total_cost': 0.0,
            'stage_costs': {},
            'call_count': 0
        }
    
    def process_query(self, question: str, proceeding: str, model: str) -> Dict[str, Any]:
        """Main entry point for query processing"""
        st.info(f"🤖 **Smart Agent Processing**: {question[:100]}...")
        
        # Step 1: Classify the query
        classification = self._classify_query(question, proceeding)
        st.write(f"📋 **Query Type**: {classification['query_type']}")
        
        # Step 2: Route to appropriate processing
        if classification['query_type'] == 'summary':
            return self._process_summary_request(question, proceeding, model, classification)
        elif classification['query_type'] == 'factual':
            return self._process_factual_query(question, proceeding, model, classification)
        elif classification['query_type'] == 'comparative':
            return self._process_comparative_query(question, proceeding, model, classification)
        else:
            return self._process_general_query(question, proceeding, model, classification)
    
    def _classify_query(self, question: str, proceeding: str = None) -> Dict[str, Any]:
        """Classify query type using LLM with stepback reasoning and document identification"""
        
        # Get available documents for analysis (filtered by proceeding if specified)
        # Use a set to track unique documents to avoid duplicates from multiple chunks
        seen_documents = set()
        available_docs = []
        
        for doc in self.documents:
            source = doc.metadata.get('source', '')
            if source in self.metadata and source not in seen_documents:
                doc_meta = self.metadata[source]
                
                # Filter by proceeding if specified
                if proceeding and proceeding != "All Proceedings":
                    doc_proceeding = doc_meta.get('proceeding', '')
                    if doc_proceeding != proceeding:
                        continue
                
                # Get relationships from metadata (analyzed during processing)
                relationships = doc_meta.get('relationships', {})
                
                available_docs.append({
                    'filename': source,
                    'document_type': doc_meta.get('document_type', 'Unknown'),
                    'filed_by': doc_meta.get('filed_by', 'Unknown'),
                    'filing_date': doc_meta.get('filing_date', 'Unknown'),
                    'description': doc_meta.get('description', '')[:200] + '...' if len(doc_meta.get('description', '')) > 200 else doc_meta.get('description', ''),
                    'relationships': relationships
                })
                
                # Mark this document as seen to avoid duplicates
                seen_documents.add(source)
        
        
        # Step 1: Stepback reasoning with document analysis (include all documents but only metadata)
        proceeding_context = f" (Proceeding: {proceeding})" if proceeding and proceeding != "All Proceedings" else ""
        
        stepback_prompt = f"""Let's step back and think about this question at a higher level within the context of a specific CPUC proceeding{proceeding_context}:

QUESTION: "{question}"

AVAILABLE DOCUMENTS IN THIS PROCEEDING ({len(available_docs)} total documents):
{self._format_documents_for_analysis(available_docs)}

Before classifying this question, let's consider:
1. What is the user fundamentally trying to understand or accomplish within this CPUC proceeding?
2. What regulatory context, parties, or issues are they asking about in this specific proceeding?
3. What type of information or analysis would best serve their needs for understanding this proceeding's documents?
4. Are they asking about originating documents (motions, proposed decisions) or responses to those documents?
5. Do they need a comprehensive overview of the proceeding or specific factual information?
6. **IMPORTANT**: If the question mentions specific document types (like "proposed decision", "motion", "application", "testimony"), focus on documents of that type. For example, if asking about a "proposed decision", consider documents with document_type containing "Decision" or "Proposed Decision". If there are multiple, focus on the most recent ones.
7. **CLASSIFICATION HINT**: Questions like "What was the [document type] here?" or "What did the [document type] say?" are asking for a SUMMARY of that specific document type, not just factual information.
8. **PARTY SUMMARY HINT**: Questions like "What are [party name]'s concerns?" or "What did [party name] say?" are asking for a SUMMARY of all documents from that party, not just factual information.
9. Which specific documents from the available list are most likely to contain relevant information for this question?

Think about the user's intent within the specific CPUC proceeding context and identify the most relevant documents for analysis. Pay special attention to document type matching and whether they want a summary or just facts. Ignore orders extending statutory deadlines"""

        stepback_system = """You are an expert regulatory analyst who helps clarify user questions about CPUC proceedings. 
        Think step by step about what the user is really asking for, considering the regulatory context and their underlying needs."""
        
        try:
            # Get stepback reasoning
            stepback_response, stepback_usage = make_openai_call(
                prompt=stepback_prompt,
                system_message=stepback_system,
                model="gpt-4o-mini",
                max_tokens=300,
                temperature=0.1,
                return_usage=True
            )
            
            # Track stepback costs
            if stepback_usage:
                cost = calculate_cost(stepback_usage.get('prompt_tokens', 0), stepback_usage.get('completion_tokens', 0), "gpt-4o-mini")
                self._track_cost('stepback_reasoning', cost)
            
            # Display stepback reasoning
            with st.expander("🧠 **Stepback Analysis**", expanded=False):
                st.write(stepback_response)
            
            # Step 2: Classification with document identification
            classification_prompt = f"""Based on this stepback analysis, classify the user's question and identify the most relevant documents:

STEPBACK ANALYSIS:
{stepback_response}

ORIGINAL QUESTION: "{question}"

CATEGORIES (within CPUC proceeding context):
1. **summary** - User wants a comprehensive overview of the proceeding, including originating documents and party responses, OR wants a detailed summary of a specific document type (e.g., "What was the proposed decision here?"), OR wants a summary of all documents from a specific party (e.g., "What are [party name]'s concerns?")
2. **factual** - User wants specific facts, dates, numbers, or precise information from the proceeding documents
3. **comparative** - User wants to compare party positions, responses, or analyze differences between submissions
4. **general** - General question about the proceeding that doesn't fit the above categories

Please respond with ONLY a JSON object in this exact format:
{{
    "query_type": "summary|factual|comparative|general",
    "complexity": "low|medium|high",
    "requires_document_chains": true|false,
    "processing_strategy": "multi_stage_summarization|direct_search|comparative_analysis|standard_qa",
    "high_priority_documents": ["filename1", "filename2", "filename3"],
    "search_priority": "high_priority_first|comprehensive_search",
    "reasoning": "Brief explanation based on the stepback analysis and proceeding context"
}}

Focus on the user's intent within the specific CPUC proceeding and identify which documents are most likely to contain relevant information."""

            system_message = """You are an expert at analyzing user questions and classifying them for optimal processing in questions related to a corpus of CPUC documents. 
            Use the stepback analysis to make a more informed classification decision.
            Be precise and consider the user's intent, not just keywords. 
            Return only valid JSON with no additional text."""
            
            # Use GPT-4-mini for classification (better reasoning, still cost-effective)
            classification_response, usage = make_openai_call(
                prompt=classification_prompt,
                system_message=system_message,
                model="gpt-4o-mini",
                max_tokens=300,
                temperature=0.1,
                return_usage=True
            )
            
            # Parse JSON response
            import json
            classification = json.loads(classification_response.strip())
            
            # Track costs for classification
            if usage:
                cost = calculate_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), "gpt-4o-mini")
                self._track_cost('classification', cost)
            
            st.write(f"🔍 **Query Classification**: {classification['query_type']} ({classification['complexity']} complexity)")
            st.write(f"💭 **Reasoning**: {classification['reasoning']}")
            
            return classification
            
        except Exception as e:
            st.warning(f"⚠️ Error in query classification: {e}. Defaulting to general query.")
            return {
                'query_type': 'general',
                'complexity': 'medium',
                'requires_document_chains': False,
                'processing_strategy': 'standard_qa',
                'reasoning': 'Classification failed, using default'
            }
    
    def _process_summary_request(self, question: str, proceeding: str, model: str, classification: Dict) -> Dict[str, Any]:
        """Process summary requests with multi-stage approach"""
        st.info("🔄 **Processing Summary Request with Multi-Stage Approach**")
        
        # Step 1: Find document chains with classification guidance
        document_chains = self._discover_document_chains(proceeding, classification, question)
        
        if not document_chains['originating_documents']:
            return {
                'answer': "No originating documents found in this proceeding.",
                'sources': [],
                'processing_stages': [],
                'cost_breakdown': self.cost_tracker
            }
        
        # Step 2: Multi-stage summarization
        return self._multi_stage_summarization(question, document_chains, model)
    
    def _discover_document_chains(self, proceeding: str, classification: Dict = None, query: str = None) -> Dict[str, Any]:
        """Discover document chains at document level using pre-analyzed relationships"""
        
        # Get unique documents (not chunks) for this proceeding
        unique_documents = {}  # filename -> document metadata
        
        # Collect unique documents from chunks
        for doc in self.documents:
            source = doc.metadata.get('source', '')
            if source in self.metadata:
                doc_meta = self.metadata[source]
                
                # Filter by proceeding if specified
                if proceeding and proceeding != "All Proceedings":
                    doc_proceeding = doc_meta.get('proceeding', '')
                    if doc_proceeding != proceeding:
                        continue
                
                # Store unique document metadata
                if source not in unique_documents:
                    unique_documents[source] = doc_meta
        
        if not unique_documents:
            st.warning("⚠️ No documents found for this proceeding.")
            return {
                'originating_documents': [],
                'response_chains': {},
                'additional_relevant_docs': [],
                'total_documents': 0,
                'total_responses': 0,
                'total_additional': 0
            }
        
        # Step 1: Find originating documents using document metadata
        originating_documents = []
        originating_types = [
            'motion', 'proposed decision', 'scoping ruling', 'scoping memo', 
            'decision', 'ruling', 'order', 'application', 'petition'
        ]
        
        for filename, doc_meta in unique_documents.items():
            doc_type = doc_meta.get('document_type', '').lower()
            
            if any(orig_type in doc_type for orig_type in originating_types):
                originating_documents.append({
                    'filename': filename,
                    'metadata': doc_meta,
                    'type': doc_type,
                    'filed_by': doc_meta.get('filed_by', 'Unknown'),
                    'filing_date': doc_meta.get('filing_date', 'Unknown')
                })
        
        # Step 2: Build response chains using pre-analyzed relationships
        response_chains = {}
        
        for orig_doc in originating_documents:
            orig_filename = orig_doc['filename']
            responses = []
            seen_responses_for_this_orig = set()  # Track unique responses for this originating document
            
            # Use pre-analyzed relationships to find responding documents
            for filename, doc_meta in unique_documents.items():
                if 'relationships' in doc_meta:
                    doc_rel = doc_meta['relationships']
                    if (doc_rel.get('responding_to') == orig_filename or 
                        orig_filename in doc_rel.get('responding_to', '')):
                        
                        # Only add if we haven't seen this response document for this originating document
                        if filename not in seen_responses_for_this_orig:
                            responses.append({
                                'filename': filename,
                                'metadata': doc_meta,
                                'relationship_type': doc_rel.get('response_type', 'Unknown')
                            })
                            seen_responses_for_this_orig.add(filename)
            
            response_chains[orig_filename] = responses
        
        # Debug: Check for duplicate responses
        total_responses = sum(len(responses) for responses in response_chains.values())
        if total_responses > 100:  # If we have too many responses, something is wrong
            st.warning(f"⚠️ High response count detected: {total_responses}. Checking for duplicates...")
            
            # Check for duplicate filenames in responses
            all_response_filenames = []
            for orig_filename, responses in response_chains.items():
                for response in responses:
                    all_response_filenames.append(response['filename'])
            
            unique_response_filenames = set(all_response_filenames)
            if len(all_response_filenames) != len(unique_response_filenames):
                st.error(f"❌ Duplicate response documents found! Total: {len(all_response_filenames)}, Unique: {len(unique_response_filenames)}")
                
                # Show duplicates
                from collections import Counter
                filename_counts = Counter(all_response_filenames)
                duplicates = {filename: count for filename, count in filename_counts.items() if count > 1}
                if duplicates:
                    st.write("**Duplicate response documents:**")
                    for filename, count in duplicates.items():
                        st.write(f"  - {filename}: {count} times")
        
        # Step 3: Add query-based document selection for additional relevance
        additional_relevant_docs = []
        if query:
            # Convert unique documents to list for query analysis
            doc_list = [{'filename': filename, 'metadata': doc_meta} 
                       for filename, doc_meta in unique_documents.items()]
            additional_relevant_docs = self._find_query_relevant_documents_document_level(query, doc_list, classification)
        
        # Display document list for transparency
        with st.expander("📋 **Documents to be Analyzed**", expanded=False):
            st.write(f"**Total Documents in Proceeding**: {len(unique_documents)}")
            st.write(f"**Originating Documents**: {len(originating_documents)}")
            
            # List originating documents
            if originating_documents:
                st.write("**Originating Documents:**")
                for i, orig_doc in enumerate(originating_documents, 1):
                    st.write(f"{i}. **{orig_doc['type']}** by {orig_doc['filed_by']} ({orig_doc['filing_date']})")
            
            # List response documents
            total_responses = sum(len(responses) for responses in response_chains.values())
            st.write(f"**Response Documents**: {total_responses}")
            
            if response_chains:
                for orig_filename, responses in response_chains.items():
                    if responses:
                        st.write(f"**Responses to {orig_filename}:**")
                        for response in responses:
                            st.write(f"  - {response['metadata'].get('document_type', 'Unknown')} by {response['metadata'].get('filed_by', 'Unknown')} ({response['metadata'].get('filing_date', 'Unknown')})")
            
            # List additional relevant documents
            if additional_relevant_docs:
                st.write(f"**Additional Query-Relevant Documents**: {len(additional_relevant_docs)}")
                for i, doc in enumerate(additional_relevant_docs, 1):
                    st.write(f"{i}. **{doc['metadata'].get('document_type', 'Unknown')}** by {doc['metadata'].get('filed_by', 'Unknown')} ({doc['metadata'].get('filing_date', 'Unknown')}) - {doc.get('relevance_reason', 'Query relevant')}")
        
        return {
            'originating_documents': originating_documents,
            'response_chains': response_chains,
            'additional_relevant_docs': additional_relevant_docs,
            'total_documents': len(unique_documents),
            'total_responses': total_responses,
            'total_additional': len(additional_relevant_docs)
        }
    
    def _find_query_relevant_documents(self, query: str, proceeding_docs: List[Document], classification: Dict = None) -> List[Dict]:
        """Find additional documents relevant to the query using semantic and keyword matching"""
        if not query:
            return []
        
        relevant_docs = []
        query_lower = query.lower()
        
        # Extract key terms from query for matching
        query_terms = set()
        for word in query_lower.split():
            if len(word) > 3:  # Skip short words
                query_terms.add(word)
        
        # Also extract potential document type keywords
        doc_type_keywords = [
            'motion', 'application', 'petition', 'complaint', 'protest', 'reply', 
            'comment', 'brief', 'testimony', 'exhibit', 'response', 'objection',
            'support', 'opposition', 'recommendation', 'proposal', 'amendment'
        ]
        
        query_doc_types = [term for term in query_terms if term in doc_type_keywords]
        
        for doc in proceeding_docs:
            source = doc.metadata.get('source', '')
            if source in self.metadata:
                doc_meta = self.metadata[source]
                doc_type = doc_meta.get('document_type', '').lower()
                description = doc_meta.get('description', '').lower()
                filed_by = doc_meta.get('filed_by', '').lower()
                
                relevance_score = 0
                relevance_reasons = []
                
                # Check for document type matches
                for query_doc_type in query_doc_types:
                    if query_doc_type in doc_type:
                        relevance_score += 3
                        relevance_reasons.append(f"Document type matches query: {query_doc_type}")
                
                # Check for keyword matches in description
                description_matches = sum(1 for term in query_terms if term in description)
                if description_matches > 0:
                    relevance_score += description_matches
                    relevance_reasons.append(f"Description contains {description_matches} query terms")
                
                # Check for party/filer matches
                for term in query_terms:
                    if term in filed_by:
                        relevance_score += 2
                        relevance_reasons.append(f"Filer matches query term: {term}")
                
                # Check for specific query patterns
                if 'response' in query_lower and 'response' in doc_type:
                    relevance_score += 2
                    relevance_reasons.append("Query asks about responses")
                elif 'comment' in query_lower and 'comment' in doc_type:
                    relevance_score += 2
                    relevance_reasons.append("Query asks about comments")
                elif 'brief' in query_lower and 'brief' in doc_type:
                    relevance_score += 2
                    relevance_reasons.append("Query asks about briefs")
                
                # Only include documents with meaningful relevance
                if relevance_score >= 2:
                    relevant_docs.append({
                        'document': doc,
                        'metadata': doc_meta,
                        'relevance_score': relevance_score,
                        'relevance_reason': '; '.join(relevance_reasons)
                    })
        
        # Sort by relevance score and return top documents
        relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_docs[:10]  # Limit to top 10 additional relevant documents
    
    def _find_query_relevant_documents_document_level(self, query: str, document_list: List[Dict], classification: Dict = None) -> List[Dict]:
        """Find additional documents relevant to the query at document level"""
        if not query:
            return []
        
        relevant_docs = []
        query_lower = query.lower()
        
        # Extract key terms from query for matching
        query_terms = set()
        for word in query_lower.split():
            if len(word) > 3:  # Skip short words
                query_terms.add(word)
        
        # Also extract potential document type keywords
        doc_type_keywords = [
            'motion', 'application', 'petition', 'complaint', 'protest', 'reply', 
            'comment', 'brief', 'testimony', 'exhibit', 'response', 'objection',
            'support', 'opposition', 'recommendation', 'proposal', 'amendment'
        ]
        
        query_doc_types = [term for term in query_terms if term in doc_type_keywords]
        
        for doc in document_list:
            doc_meta = doc['metadata']
            doc_type = doc_meta.get('document_type', '').lower()
            description = doc_meta.get('description', '').lower()
            filed_by = doc_meta.get('filed_by', '').lower()
            
            relevance_score = 0
            relevance_reasons = []
            
            # Check for document type matches
            for query_doc_type in query_doc_types:
                if query_doc_type in doc_type:
                    relevance_score += 3
                    relevance_reasons.append(f"Document type matches query: {query_doc_type}")
            
            # Check for keyword matches in description
            description_matches = sum(1 for term in query_terms if term in description)
            if description_matches > 0:
                relevance_score += description_matches
                relevance_reasons.append(f"Description contains {description_matches} query terms")
            
            # Check for party/filer matches
            for term in query_terms:
                if term in filed_by:
                    relevance_score += 2
                    relevance_reasons.append(f"Filer matches query term: {term}")
            
            # Check for specific query patterns
            if 'response' in query_lower and 'response' in doc_type:
                relevance_score += 2
                relevance_reasons.append("Query asks about responses")
            elif 'comment' in query_lower and 'comment' in doc_type:
                relevance_score += 2
                relevance_reasons.append("Query asks about comments")
            elif 'brief' in query_lower and 'brief' in doc_type:
                relevance_score += 2
                relevance_reasons.append("Query asks about briefs")
            
            # Only include documents with meaningful relevance
            if relevance_score >= 2:
                relevant_docs.append({
                    'filename': doc['filename'],
                    'metadata': doc_meta,
                    'relevance_score': relevance_score,
                    'relevance_reason': '; '.join(relevance_reasons)
                })
        
        # Sort by relevance score and return top documents
        relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_docs[:10]  # Limit to top 10 additional relevant documents
    
    def _summarize_additional_documents(self, additional_docs: List[Dict], model: str) -> List[Dict]:
        """Summarize additional relevant documents"""
        if not additional_docs:
            return []
        
        summaries = []
        
        for doc_info in additional_docs:
            doc = doc_info['document']
            metadata = doc_info['metadata']
            relevance_reason = doc_info.get('relevance_reason', 'Query relevant')
            
            # Create summary prompt
            summary_prompt = f"""
            Summarize this document focusing on its relevance to the query context.
            
            Document Type: {metadata.get('document_type', 'Unknown')}
            Filed By: {metadata.get('filed_by', 'Unknown')}
            Filing Date: {metadata.get('filing_date', 'Unknown')}
            Relevance Reason: {relevance_reason}
            
            Document Content:
            {doc.page_content[:4000]}  # Limit content to avoid token limits
            
            Provide a concise summary highlighting:
            1. Key points and arguments
            2. How it relates to the query context
            3. Important details or evidence
            4. Any notable positions or recommendations
            
            Keep the summary focused and under 500 words.
            """
            
            try:
                from src.utils.llm_utils import make_openai_call, calculate_cost
                
                response, usage = make_openai_call(
                    prompt=summary_prompt,
                    model=model,
                    max_tokens=400,
                    temperature=0.3,
                    return_usage=True
                )
                
                # Track costs
                if usage:
                    cost = calculate_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), model)
                    self._track_cost('stage_3', cost)
                
                summaries.append({
                    'document_type': metadata.get('document_type', 'Unknown'),
                    'filed_by': metadata.get('filed_by', 'Unknown'),
                    'filing_date': metadata.get('filing_date', 'Unknown'),
                    'relevance_reason': relevance_reason,
                    'summary': response.strip(),
                    'source': metadata.get('filename', 'Unknown'),
                    'page_numbers': doc.metadata.get('page_numbers', [])
                })
                
            except Exception as e:
                st.warning(f"⚠️ Error summarizing additional document {metadata.get('filename', 'Unknown')}: {e}")
                continue
        
        return summaries
    
    def _multi_stage_summarization(self, question: str, document_chains: Dict, model: str) -> Dict[str, Any]:
        """Multi-stage summarization with detailed extraction and page citations"""
        # Removed verbose output
        
        # Display analysis plan
        with st.expander("📋 **Analysis Plan**", expanded=True):
            st.write(f"**Question**: {question}")
            st.write(f"**Originating Documents to Summarize**: {len(document_chains['originating_documents'])}")
            st.write(f"**Response Documents to Summarize**: {document_chains.get('total_responses', 0)}")
            st.write(f"**Additional Relevant Documents**: {document_chains.get('total_additional', 0)}")
            st.write(f"**Model Used**: {model}")
            st.write("**Process**: 1) Summarize originating documents → 2) Summarize responses → 3) Summarize additional relevant docs → 4) Comparative analysis → 5) Overall synthesis")
        
        processing_stages = []
        all_summaries = []
        stage_outputs = {}
        
        # Stage 1: Summarize originating documents
        st.write("**Stage 1: Originating Document Summaries**")
        originating_summaries = self._summarize_originating_documents(
            document_chains['originating_documents'], model
        )
        
        # Display Stage 1 results in expandable section
        with st.expander(f"📄 **Stage 1 Results: Originating Document Summaries** (Model: {model})", expanded=False):
            st.write(f"**Documents Processed**: {len(originating_summaries)}")
            st.write(f"**Model Used**: {model}")
            st.write(f"**Cost**: ${self.cost_tracker['stage_costs'].get('stage_1', 0):.4f}")
            
            for i, summary in enumerate(originating_summaries, 1):
                st.write(f"**{i}. {summary['document_type']} by {summary['filed_by']}**")
                st.write(summary['summary'])
                st.divider()
        
        processing_stages.append({
            'stage': 1,
            'description': 'Originating Document Summaries',
            'documents_processed': len(originating_summaries),
            'cost': self.cost_tracker['stage_costs'].get('stage_1', 0),
            'model': model,
            'output': originating_summaries
        })
        all_summaries.extend(originating_summaries)
        stage_outputs['stage_1'] = originating_summaries
        
        # Stage 2: Summarize response documents
        st.write("**Stage 2: Response Document Summaries**")
        response_summaries = self._summarize_response_documents(
            document_chains['response_chains'], model
        )
        
        # Display Stage 2 results in expandable section
        with st.expander(f"📝 **Stage 2 Results: Response Document Summaries** (Model: {model})", expanded=False):
            st.write(f"**Documents Processed**: {len(response_summaries)}")
            st.write(f"**Model Used**: {model}")
            st.write(f"**Cost**: ${self.cost_tracker['stage_costs'].get('stage_2', 0):.4f}")
            
            for i, summary in enumerate(response_summaries, 1):
                st.write(f"**{i}. {summary['document_type']} by {summary['filed_by']}**")
                st.write(f"*Response to: {summary['response_to']}*")
                st.write(summary['summary'])
                st.divider()
        
        processing_stages.append({
            'stage': 2,
            'description': 'Response Document Summaries',
            'documents_processed': len(response_summaries),
            'cost': self.cost_tracker['stage_costs'].get('stage_2', 0),
            'model': model,
            'output': response_summaries
        })
        all_summaries.extend(response_summaries)
        stage_outputs['stage_2'] = response_summaries
        
        # Stage 3: Summarize additional relevant documents
        additional_summaries = []
        if document_chains.get('additional_relevant_docs'):
            st.write("**Stage 3: Additional Relevant Document Summaries**")
            additional_summaries = self._summarize_additional_documents(
                document_chains['additional_relevant_docs'], model
            )
            
            # Display Stage 3 results in expandable section
            with st.expander(f"📋 **Stage 3 Results: Additional Relevant Document Summaries** (Model: {model})", expanded=False):
                st.write(f"**Documents Processed**: {len(additional_summaries)}")
                st.write(f"**Model Used**: {model}")
                st.write(f"**Cost**: ${self.cost_tracker['stage_costs'].get('stage_3', 0):.4f}")
                
                for i, summary in enumerate(additional_summaries, 1):
                    st.write(f"**{i}. {summary['document_type']} by {summary['filed_by']}**")
                    st.write(f"*Relevance: {summary['relevance_reason']}*")
                    st.write(summary['summary'])
                    st.divider()
            
            processing_stages.append({
                'stage': 3,
                'description': 'Additional Relevant Document Summaries',
                'documents_processed': len(additional_summaries),
                'cost': self.cost_tracker['stage_costs'].get('stage_3', 0),
                'model': model,
                'output': additional_summaries
            })
            all_summaries.extend(additional_summaries)
            stage_outputs['stage_3'] = additional_summaries
        
        # Stage 4: Comparative analysis
        st.write("**Stage 4: Comparative Analysis**")
        comparative_analysis = self._create_comparative_analysis(
            document_chains, model
        )
        
        # Display Stage 4 results in expandable section
        with st.expander(f"🔍 **Stage 4 Results: Comparative Analysis** (Model: {model})", expanded=False):
            st.write(f"**Model Used**: {model}")
            st.write(f"**Cost**: ${self.cost_tracker['stage_costs'].get('stage_4', 0):.4f}")
            st.write("**Comparative Analysis:**")
            st.write(comparative_analysis)
        
        processing_stages.append({
            'stage': 4,
            'description': 'Comparative Analysis',
            'analysis_created': True,
            'cost': self.cost_tracker['stage_costs'].get('stage_4', 0),
            'model': model,
            'output': comparative_analysis
        })
        all_summaries.append({'type': 'comparative_analysis', 'content': comparative_analysis})
        stage_outputs['stage_4'] = comparative_analysis
        
        # Stage 5: Overall synthesis
        st.write("**Stage 5: Overall Synthesis**")
        final_synthesis = self._create_overall_synthesis(
            question, all_summaries, model
        )
        
        # Display Stage 5 results in expandable section
        with st.expander(f"🎯 **Stage 5 Results: Overall Synthesis** (Model: {model})", expanded=False):
            st.write(f"**Model Used**: {model}")
            st.write(f"**Cost**: ${self.cost_tracker['stage_costs'].get('stage_5', 0):.4f}")
            st.write("**Final Synthesis:**")
            st.write(final_synthesis)
        
        processing_stages.append({
            'stage': 5,
            'description': 'Overall Synthesis',
            'synthesis_created': True,
            'cost': self.cost_tracker['stage_costs'].get('stage_5', 0),
            'model': model,
            'output': final_synthesis
        })
        stage_outputs['stage_5'] = final_synthesis
        
        # Display overall processing summary
        with st.expander("📊 **Processing Summary**", expanded=True):
            st.write(f"**Total Cost**: ${self.cost_tracker['total_cost']:.4f}")
            st.write(f"**Total API Calls**: {self.cost_tracker['call_count']}")
            st.write(f"**Model Used**: {model}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stage 1", f"${self.cost_tracker['stage_costs'].get('stage_1', 0):.4f}")
            with col2:
                st.metric("Stage 2", f"${self.cost_tracker['stage_costs'].get('stage_2', 0):.4f}")
            with col3:
                st.metric("Stage 3", f"${self.cost_tracker['stage_costs'].get('stage_3', 0):.4f}")
            with col4:
                st.metric("Stage 4", f"${self.cost_tracker['stage_costs'].get('stage_4', 0):.4f}")
        
        # Prepare sources
        sources = self._prepare_comprehensive_sources(document_chains)
        
        return {
            'answer': final_synthesis,
            'sources': sources,
            'processing_stages': processing_stages,
            'cost_breakdown': self.cost_tracker,
            'detailed_summaries': all_summaries,
            'stage_outputs': stage_outputs
        }
    
    def _summarize_originating_documents(self, originating_docs: List[Dict], model: str) -> List[Dict[str, Any]]:
        """Summarize originating documents using only large chunks"""
        summaries = []
        
        for i, orig_doc in enumerate(originating_docs, 1):
            st.write(f"📄 **Summarizing Originating Document {i}/{len(originating_docs)}: {orig_doc['type']}**")
            
            # Get large chunks for this document
            large_chunks = self._get_large_chunks_for_document(orig_doc['filename'])
            
            if not large_chunks:
                st.warning(f"⚠️ No large chunks found for {orig_doc['filename']}")
                continue
            
            # Combine large chunks with page number preservation
            combined_content_parts = []
            all_page_numbers = set()
            
            for chunk in large_chunks:
                chunk_content = chunk.page_content
                chunk_pages = chunk.metadata.get('page_numbers', [])
                all_page_numbers.update(chunk_pages)
                
                # Add page number markers if they exist
                if chunk_pages:
                    page_marker = f" [PAGES {', '.join(map(str, sorted(chunk_pages)))}]"
                    combined_content_parts.append(f"{chunk_content}{page_marker}")
                else:
                    combined_content_parts.append(chunk_content)
            
            doc_content = "\n\n".join(combined_content_parts)
            
            # Calculate dynamic max tokens based on model and content length
            max_tokens = self._get_dynamic_max_tokens(model, doc_content)
            
            # Create detailed summarization prompt with page number emphasis
            page_info = ""
            if all_page_numbers:
                page_info = f"\n- Available Pages: {', '.join(map(str, sorted(all_page_numbers)))}"
            
            prompt = f"""Please provide a comprehensive summary of this {orig_doc['type']} with maximum detail and specific page citations.

DOCUMENT INFORMATION:
- Type: {orig_doc['type']}
- Filed by: {orig_doc['filed_by']}
- Date: {orig_doc['filing_date']}
- Source: {orig_doc['filename']}{page_info}

DOCUMENT CONTENT:
{doc_content}

Please provide a detailed summary that includes:
1. **Executive Summary** (2-3 sentences with page citations)
2. **Key Arguments and Positions** (with specific page citations)
3. **Regulatory Issues Addressed** (with page references)
4. **Legal Conclusions and Holdings** (with page citations)
5. **Impact and Implications** (with supporting page references)
6. **Specific Recommendations or Requests** (with page citations)

IMPORTANT: This document contains page number markers [PAGES X, Y, Z] throughout the content. Use these page numbers in your citations. For each point, include specific page numbers where the information can be found. Use format: "According to page X..." or "As stated on page Y..."

Focus on extracting the maximum amount of detail while maintaining accuracy and providing clear page citations."""

            system_message = """You are an expert regulatory analyst specializing in CPUC proceedings. 
            Your task is to extract maximum detail from regulatory documents while providing specific page citations. 
            Be thorough, accurate, and always cite page numbers for key information. 
            Focus on regulatory significance, legal arguments, and practical implications."""
            
            try:
                summary_content, usage = make_openai_call(
                    prompt=prompt,
                    system_message=system_message,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.1,
                    return_usage=True
                )
                
                # Track costs
                if usage:
                    cost = calculate_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), model)
                    self._track_cost('stage_1', cost)
                
                summaries.append({
                    'document_type': orig_doc['type'],
                    'filed_by': orig_doc['filed_by'],
                    'summary': summary_content,
                    'metadata': orig_doc['metadata']
                })
                
                st.success(f"✅ Summarized {orig_doc['type']} by {orig_doc['filed_by']}")
                
            except Exception as e:
                st.error(f"❌ Error summarizing {orig_doc['type']}: {e}")
                summaries.append({
                    'document_type': orig_doc['type'],
                    'filed_by': orig_doc['filed_by'],
                    'summary': f"Error generating summary: {e}",
                    'metadata': orig_doc['metadata']
                })
        
        return summaries
    
    def _get_large_chunks_for_document(self, filename: str) -> List[Document]:
        """Get only large chunks for a specific document"""
        large_chunks = []
        
        for doc in self.documents:
            source = doc.metadata.get('source', '')
            chunk_level = doc.metadata.get('chunk_level', '')
            
            if source == filename and chunk_level == 'large':
                large_chunks.append(doc)
        
        return large_chunks
    
    def _summarize_response_documents(self, response_chains: Dict, model: str) -> List[Dict[str, Any]]:
        """Summarize response documents using only large chunks"""
        summaries = []
        
        for orig_filename, responses in response_chains.items():
            if not responses:
                continue
                
            st.write(f"📝 **Summarizing {len(responses)} response documents to {orig_filename}**")
            
            # Process each response document using large chunks only
            for response in responses:
                filename = response['filename']
                doc_meta = response['metadata']
                
                # Get large chunks for this response document
                large_chunks = self._get_large_chunks_for_document(filename)
                
                if not large_chunks:
                    st.warning(f"⚠️ No large chunks found for {filename}")
                    continue
                
                # Combine large chunks with page number preservation
                combined_content_parts = []
                all_page_numbers = set()
                
                for chunk in large_chunks:
                    chunk_content = chunk.page_content
                    chunk_pages = chunk.metadata.get('page_numbers', [])
                    all_page_numbers.update(chunk_pages)
                    
                    # Add page number markers if they exist
                    if chunk_pages:
                        page_marker = f" [PAGES {', '.join(map(str, sorted(chunk_pages)))}]"
                        combined_content_parts.append(f"{chunk_content}{page_marker}")
                    else:
                        combined_content_parts.append(chunk_content)
                
                combined_content = "\n\n".join(combined_content_parts)
                
                # Add page number information
                page_info = ""
                if all_page_numbers:
                    page_info = f"\n- Available Pages: {', '.join(map(str, sorted(all_page_numbers)))}"
                
                prompt = f"""Please provide a detailed summary of this response document with specific page citations and party position analysis.

RESPONSE DOCUMENT INFORMATION:
- Type: {doc_meta.get('document_type', 'Unknown')}
- Filed by: {doc_meta.get('filed_by', 'Unknown')}
- Date: {doc_meta.get('filing_date', 'Unknown')}
- Source: {filename}
- Response to: {orig_filename}{page_info}

DOCUMENT CONTENT:
{combined_content}

Please provide a comprehensive summary that includes:
1. **Party Position Summary** (2-3 sentences with page citations)
2. **Key Arguments and Stance** (with specific page references)
3. **Areas of Agreement/Disagreement** (with page citations)
4. **Specific Concerns or Support** (with page references)
5. **Legal Arguments Presented** (with page citations)
6. **Recommendations or Requests** (with page references)

For each point, include specific page numbers where the information can be found. 
Focus on the party's specific position and arguments with detailed citations."""

                system_message = """You are an expert regulatory analyst specializing in party submissions and responses in CPUC proceedings. 
                Your task is to extract detailed party positions, arguments, and stances while providing specific page citations. 
                Focus on what the party is arguing for or against, their specific concerns, and their recommendations. 
                Always cite page numbers for key information."""
                
                try:
                    # Calculate dynamic max tokens
                    max_tokens = self._get_dynamic_max_tokens(model, combined_content)
                    
                    summary_content, usage = make_openai_call(
                        prompt=prompt,
                        system_message=system_message,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=0.1,
                        return_usage=True
                    )
                    
                    # Track costs
                    if usage:
                        cost = calculate_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), model)
                        self._track_cost('stage_2', cost)
                    
                    summaries.append({
                        'document_type': doc_meta.get('document_type', 'Unknown'),
                        'filed_by': doc_meta.get('filed_by', 'Unknown'),
                        'summary': summary_content,
                        'metadata': doc_meta,
                        'response_to': orig_filename
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error summarizing response: {e}")
                    summaries.append({
                        'document_type': doc_meta.get('document_type', 'Unknown'),
                        'filed_by': doc_meta.get('filed_by', 'Unknown'),
                        'summary': f"Error generating summary: {e}",
                        'metadata': doc_meta,
                        'response_to': orig_filename
                    })
        
        return summaries
    
    def _create_comparative_analysis(self, document_chains: Dict, model: str) -> str:
        """Create comparative analysis of responses"""
        st.write("🔍 **Creating Comparative Analysis**")
        
        # Prepare comparative data
        comparative_data = []
        for orig_filename, responses in document_chains['response_chains'].items():
            if responses:
                response_summaries = []
                for response in responses:
                    doc_meta = response['document'].metadata
                    if doc_meta.get('source', '') in self.metadata:
                        full_meta = self.metadata[doc_meta.get('source', '')]
                    else:
                        full_meta = doc_meta
                    
                    response_summaries.append({
                        'party': full_meta.get('filed_by', 'Unknown'),
                        'type': full_meta.get('document_type', 'Unknown'),
                        'content': response['document'].page_content[:1000]  # Truncate for prompt
                    })
                
                comparative_data.append({
                    'originating_document': orig_filename,
                    'responses': response_summaries
                })
        
        if not comparative_data:
            return "No response documents found for comparative analysis."
        
        # Create comparative analysis prompt
        prompt = f"""Please create a comprehensive comparative analysis of party responses with specific page citations.

RESPONSE DATA:
{json.dumps(comparative_data, indent=2)}

Please provide a detailed comparative analysis that includes:
1. **Overall Response Summary** (2-3 sentences)
2. **Areas of Agreement Between Parties** (with page citations)
3. **Key Areas of Disagreement** (with page citations and party positions)
4. **Common Concerns Raised** (with page references)
5. **Divergent Positions** (with specific party stances and page citations)
6. **Regulatory Impact Assessment** (with supporting page references)

For each point, include specific page numbers and party names. 
Focus on identifying patterns of agreement and disagreement across all responding parties."""

        system_message = """You are an expert regulatory analyst specializing in comparative analysis of party responses in CPUC proceedings. 
        Your task is to identify patterns of agreement and disagreement between parties, common concerns, and divergent positions. 
        Always cite specific page numbers and party names. Focus on regulatory significance and practical implications."""
        
        try:
            # Calculate dynamic max tokens for comparative analysis
            max_tokens = self._get_dynamic_max_tokens(model, json.dumps(comparative_data))
            
            analysis, usage = make_openai_call(
                prompt=prompt,
                system_message=system_message,
                model=model,
                max_tokens=max_tokens,
                temperature=0.1,
                return_usage=True
            )
            
            # Track costs
            if usage:
                cost = calculate_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), model)
                self._track_cost('stage_3', cost)
            
            return analysis
            
        except Exception as e:
            st.error(f"❌ Error creating comparative analysis: {e}")
            return f"Error creating comparative analysis: {e}"
    
    def _create_overall_synthesis(self, question: str, all_summaries: List, model: str) -> str:
        """Create overall synthesis of all summaries"""
        st.write("🎯 **Creating Overall Synthesis**")
        
        # Prepare synthesis data
        synthesis_data = {
            'original_question': question,
            'summaries': all_summaries
        }
        
        prompt = f"""Please create a comprehensive executive summary that synthesizes all the detailed summaries provided.

ORIGINAL QUESTION: {question}

DETAILED SUMMARIES:
{json.dumps([s for s in all_summaries if isinstance(s, dict)], indent=2, default=str)}

Please provide a comprehensive executive summary that includes:
1. **Executive Overview** (3-4 sentences summarizing the key findings)
2. **Originating Document Summary** (key points from motions, proposed decisions, etc.)
3. **Party Response Analysis** (areas of agreement/disagreement with specific party positions)
4. **Key Regulatory Issues** (with page citations)
5. **Consensus and Divergence** (where parties agree vs. disagree)
6. **Regulatory Implications** (potential impact and next steps)

For each point, include specific page numbers and party names where relevant. 
This should be a comprehensive, executive-level summary that captures all the key information from the detailed analysis."""

        system_message = """You are an expert regulatory analyst creating an executive summary for CPUC proceedings. 
        Your task is to synthesize detailed party positions, regulatory issues, and areas of agreement/disagreement into a comprehensive overview. 
        Always cite specific page numbers and party names. Focus on regulatory significance and practical implications for decision-makers."""
        
        try:
            # Calculate dynamic max tokens for overall synthesis
            max_tokens = self._get_dynamic_max_tokens(model, json.dumps([s for s in all_summaries if isinstance(s, dict)], default=str))
            
            synthesis, usage = make_openai_call(
                prompt=prompt,
                system_message=system_message,
                model=model,
                max_tokens=max_tokens,
                temperature=0.1,
                return_usage=True
            )
            
            # Track costs
            if usage:
                cost = calculate_cost(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0), model)
                self._track_cost('stage_4', cost)
            
            return synthesis
            
        except Exception as e:
            st.error(f"❌ Error creating overall synthesis: {e}")
            return f"Error creating overall synthesis: {e}"
    
    def _prepare_comprehensive_sources(self, document_chains: Dict) -> List[Dict[str, Any]]:
        """Prepare comprehensive source information"""
        sources = []
        
        # Add originating documents
        for orig_doc in document_chains['originating_documents']:
            sources.append({
                'type': 'originating',
                'document_type': orig_doc['type'],
                'filed_by': orig_doc['filed_by'],
                'filing_date': orig_doc['filing_date'],
                'source': orig_doc['metadata'].get('filename', 'Unknown'),
                'proceeding': orig_doc['metadata'].get('proceeding', 'Unknown')
            })
        
        # Add response documents
        for orig_filename, responses in document_chains['response_chains'].items():
            for response in responses:
                doc_meta = response['document'].metadata
                if doc_meta.get('source', '') in self.metadata:
                    full_meta = self.metadata[doc_meta.get('source', '')]
                else:
                    full_meta = doc_meta
                
                sources.append({
                    'type': 'response',
                    'document_type': full_meta.get('document_type', 'Unknown'),
                    'filed_by': full_meta.get('filed_by', 'Unknown'),
                    'filing_date': full_meta.get('filing_date', 'Unknown'),
                    'source': full_meta.get('filename', 'Unknown'),
                    'proceeding': full_meta.get('proceeding', 'Unknown'),
                    'response_to': orig_filename
                })
        
        return sources
    
    def _track_cost(self, stage: str, cost: float):
        """Track costs by stage"""
        if stage not in self.cost_tracker['stage_costs']:
            self.cost_tracker['stage_costs'][stage] = 0.0
        self.cost_tracker['stage_costs'][stage] += cost
        self.cost_tracker['total_cost'] += cost
        self.cost_tracker['call_count'] += 1
    
    def _get_dynamic_max_tokens(self, model: str, content: str) -> int:
        """Calculate dynamic max tokens based on model context window and content length"""
        # Get model context limits
        context_limits = self._get_model_context_limits(model)
        max_context_tokens = context_limits['max_context_tokens']
        reserved_tokens = context_limits['reserved_tokens']
        
        # Calculate content length in tokens (rough estimate: 4 chars per token)
        content_tokens = len(content) // 4
        
        # Calculate available tokens for response
        available_tokens = max_context_tokens - content_tokens - reserved_tokens
        
        # Ensure we have reasonable response length
        max_tokens = min(available_tokens, max_context_tokens // 4)
        max_tokens = max(max_tokens, 500)  # Minimum response length
        
        # If content is too long, truncate it
        if content_tokens > max_context_tokens * 0.75:  # Use 75% of context for content
            max_content_tokens = int(max_context_tokens * 0.75)
            max_content_length = max_content_tokens * 4  # Convert back to characters
            content = content[:max_content_length]
            st.warning(f"⚠️ Content truncated to fit model context window ({model})")
        
        # Return max tokens for response
        return min(4000, max_tokens)
    
    def _get_dynamic_num_results(self, model: str, question: str, total_docs: int) -> int:
        """Calculate dynamic number of results based on model context window and available tokens"""
        # Get model context window limits
        context_limits = self._get_model_context_limits(model)
        max_context_tokens = context_limits['max_context_tokens']
        reserved_tokens = context_limits['reserved_tokens']
        available_tokens = max_context_tokens - reserved_tokens
        
        # Estimate tokens per document (rough approximation)
        # Average document chunk is ~500-1000 tokens, we'll use 750 as baseline
        estimated_tokens_per_doc = 750
        
        # Calculate how many documents we can fit
        max_docs_by_context = available_tokens // estimated_tokens_per_doc
        
        # Adjust based on question complexity
        question_lower = question.lower()
        complexity_indicators = [
            'compare', 'comparison', 'difference', 'versus', 'vs', 'contrast',
            'analyze', 'analysis', 'evaluate', 'assessment', 'review',
            'comprehensive', 'detailed', 'thorough', 'complete'
        ]
        
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in question_lower)
        
        # Adjust for complexity
        if complexity_score >= 3:
            max_docs_by_context = int(max_docs_by_context * 1.5)
        elif complexity_score >= 1:
            max_docs_by_context = int(max_docs_by_context * 1.2)
        
        # Don't exceed available documents or reasonable limits
        return min(max_docs_by_context, total_docs, 20)  # Cap at 20 for performance
    
    def _get_model_context_limits(self, model: str) -> Dict[str, int]:
        """Get context window limits for different models"""
        # Model context window limits (in tokens)
        context_limits = {
            'gpt-3.5-turbo': {'max_context_tokens': 4096, 'reserved_tokens': 1000},
            'gpt-3.5-turbo-16k': {'max_context_tokens': 16384, 'reserved_tokens': 2000},
            'gpt-4': {'max_context_tokens': 8192, 'reserved_tokens': 1500},
            'gpt-4-32k': {'max_context_tokens': 32768, 'reserved_tokens': 3000},
            'gpt-4-turbo': {'max_context_tokens': 128000, 'reserved_tokens': 4000},
            'gpt-4o': {'max_context_tokens': 128000, 'reserved_tokens': 4000},
            'gpt-4o-mini': {'max_context_tokens': 128000, 'reserved_tokens': 3000},
            'o1': {'max_context_tokens': 200000, 'reserved_tokens': 5000},
            'o1-mini': {'max_context_tokens': 200000, 'reserved_tokens': 4000},
            'o1-pro': {'max_context_tokens': 200000, 'reserved_tokens': 5000},
            'o3': {'max_context_tokens': 200000, 'reserved_tokens': 5000},
            'o3-mini': {'max_context_tokens': 200000, 'reserved_tokens': 4000}
        }
        
        return context_limits.get(model, {'max_context_tokens': 8000, 'reserved_tokens': 2000})
    
    def _format_documents_for_analysis(self, documents: List[Dict]) -> str:
        """Format documents for stepback analysis with relationship information"""
        if not documents:
            return "No documents available"
        
        # Group documents by type for better analysis
        by_type = {}
        for doc in documents:
            doc_type = doc['document_type']
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(doc)
        
        formatted = []
        for doc_type, docs in by_type.items():
            formatted.append(f"**{doc_type.upper()} DOCUMENTS:**")
            for i, doc in enumerate(docs, 1):
                formatted.append(f"  {i}. **{doc['filename']}**")
                formatted.append(f"     - Filed by: {doc['filed_by']}")
                formatted.append(f"     - Date: {doc['filing_date']}")
                if doc['description']:
                    formatted.append(f"     - Description: {doc['description']}")
                
                # Add relationship information if available
                if 'relationships' in doc:
                    relationships = doc['relationships']
                    if relationships:
                        formatted.append(f"     - Relationships:")
                        for rel_type, rel_info in relationships.items():
                            if rel_info:
                                formatted.append(f"       • {rel_type}: {rel_info}")
                
                formatted.append("")
            formatted.append("")
        
        return "\n".join(formatted)
    
    
    def _simple_text_search(self, question: str, documents: List[Document], model: str, num_results: int) -> Dict[str, Any]:
        """Simple text search fallback when vector/BM25 search is not available"""
        st.info("🔍 **Using Simple Text Search**")
        
        # Simple keyword matching
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        scored_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            # Count word matches
            matches = sum(1 for word in question_words if word in content_lower)
            if matches > 0:
                scored_docs.append((doc, matches))
        
        # Sort by match count and take top results
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = [doc for doc, score in scored_docs[:num_results]]
        
        if not relevant_docs:
            return {
                'answer': "No relevant documents found for your question.",
                'sources': [],
                'processing_stages': [{'stage': 'simple_search', 'description': 'Simple text search fallback'}],
                'cost_breakdown': self.cost_tracker
            }
        
        # Generate answer using LLM
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""Based on the following context from CPUC regulatory documents, please answer the question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context. If the context doesn't contain enough information to answer the question, please say so."""
        
        try:
            response_content, usage = make_openai_call(
                prompt=prompt,
                system_message="You are an expert California Public Utilities Commission (CPUC) regulatory analyst. Provide accurate analysis based on the provided context.",
                model=model,
                max_tokens=2000,
                temperature=0.1,
                return_usage=True
            )
            
            # Track costs
            if usage:
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                cost = calculate_cost(input_tokens, output_tokens, model)
                self._track_cost('simple_search', cost)
            
            # Prepare sources
            sources = []
            for doc in relevant_docs:
                source = doc.metadata.get('source', 'Unknown')
                if source in self.metadata:
                    doc_meta = self.metadata[source]
                    sources.append({
                        'source': source,
                        'document_type': doc_meta.get('document_type', 'Unknown'),
                        'filed_by': doc_meta.get('filed_by', 'Unknown'),
                        'filing_date': doc_meta.get('filing_date', 'Unknown'),
                        'relevance_score': 'Simple text match'
                    })
            
            return {
                'answer': response_content,
                'sources': sources,
                'processing_stages': [{'stage': 'simple_search', 'description': f'Simple text search with {len(relevant_docs)} results', 'cost': self.cost_tracker['stage_costs'].get('simple_search', 0)}],
                'cost_breakdown': self.cost_tracker
            }
            
        except Exception as e:
            st.error(f"❌ Error in simple text search: {e}")
            return {
                'answer': f"Error processing query: {e}",
                'sources': [],
                'processing_stages': [{'stage': 'simple_search', 'description': 'Error in simple search'}],
                'cost_breakdown': self.cost_tracker
            }
    
    def _simple_text_search_with_priority(self, question: str, documents: List[Document], model: str, num_results: int, classification: Dict) -> Dict[str, Any]:
        """Simple text search with priority document weighting"""
        st.info("🔍 **Using Simple Text Search with Priority Documents**")
        
        # Get high priority documents from classification
        high_priority_docs = classification.get('high_priority_documents', [])
        
        # Simple keyword matching with priority weighting
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        scored_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            # Count word matches
            matches = sum(1 for word in question_words if word in content_lower)
            if matches > 0:
                # Check if this is a high priority document
                source = doc.metadata.get('source', '')
                priority_boost = 2.0 if source in high_priority_docs else 1.0
                
                # Apply priority boost to score
                final_score = matches * priority_boost
                scored_docs.append((doc, final_score, source in high_priority_docs))
        
        # Sort by score and take top results
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = [doc for doc, score, is_priority in scored_docs[:num_results]]
        
        # Show priority document info
        priority_found = [source for doc, score, is_priority in scored_docs[:num_results] if is_priority]
        if priority_found:
            st.info(f"🎯 **Priority Documents Found**: {', '.join(priority_found)}")
        
        if not relevant_docs:
            return {
                'answer': "No relevant documents found for your question.",
                'sources': [],
                'processing_stages': [{'stage': 'priority_search', 'description': 'Simple text search with priority weighting'}],
                'cost_breakdown': self.cost_tracker
            }
        
        # Generate answer using LLM
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""Based on the following context from CPUC regulatory documents, please answer the question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context. If the context doesn't contain enough information to answer the question, please say so."""
        
        try:
            response_content, usage = make_openai_call(
                prompt=prompt,
                system_message="You are an expert California Public Utilities Commission (CPUC) regulatory analyst. Provide accurate analysis based on the provided context.",
                model=model,
                max_tokens=2000,
                temperature=0.1,
                return_usage=True
            )
            
            # Track costs
            if usage:
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                cost = calculate_cost(input_tokens, output_tokens, model)
                self._track_cost('priority_search', cost)
            
            # Prepare sources
            sources = []
            for doc in relevant_docs:
                source = doc.metadata.get('source', 'Unknown')
                if source in self.metadata:
                    doc_meta = self.metadata[source]
                    sources.append({
                        'source': source,
                        'document_type': doc_meta.get('document_type', 'Unknown'),
                        'filed_by': doc_meta.get('filed_by', 'Unknown'),
                        'filing_date': doc_meta.get('filing_date', 'Unknown'),
                        'relevance_score': 'Priority document' if source in high_priority_docs else 'Simple text match'
                    })
            
            return {
                'answer': response_content,
                'sources': sources,
                'processing_stages': [{'stage': 'priority_search', 'description': f'Simple text search with {len(priority_found)} priority documents', 'cost': self.cost_tracker['stage_costs'].get('priority_search', 0)}],
                'cost_breakdown': self.cost_tracker
            }
            
        except Exception as e:
            st.error(f"❌ Error in priority text search: {e}")
            return {
                'answer': f"Error processing query: {e}",
                'sources': [],
                'processing_stages': [{'stage': 'priority_search', 'description': 'Error in priority search'}],
                'cost_breakdown': self.cost_tracker
            }
    
    def _format_documents_for_analysis(self, documents: List[Dict]) -> str:
        """Format documents for analysis in a concise way"""
        if not documents:
            return "No documents available"
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            doc_type = doc.get('document_type', 'Unknown')
            filed_by = doc.get('filed_by', 'Unknown')
            filing_date = doc.get('filing_date', 'Unknown')
            relationships = doc.get('relationships', {})
            
            # Format relationships info concisely
            rel_info = ""
            if relationships:
                doc_role = relationships.get('document_role', '')
                if doc_role:
                    rel_info = f" [{doc_role}]"
            
            # Keep it concise - just essential info
            formatted_docs.append(f"{i}. {doc_type} by {filed_by} ({filing_date}){rel_info}")
        
        return "\n".join(formatted_docs)
    
    def _process_factual_query(self, question: str, proceeding: str, model: str, classification: Dict) -> Dict[str, Any]:
        """Process factual queries using vector/BM25 search with dynamic context management"""
        st.info("🔍 **Processing Factual Query with Vector/BM25 Search**")
        
        # Filter documents by proceeding - check both metadata and source
        proceeding_docs = []
        for doc in self.documents:
            # Check proceeding in document metadata
            doc_proceeding = doc.metadata.get('proceeding', '')
            # Also check if source document has proceeding info
            source = doc.metadata.get('source', '')
            if source in self.metadata:
                source_proceeding = self.metadata[source].get('proceeding', '')
            else:
                source_proceeding = ''
            
            if doc_proceeding == proceeding or source_proceeding == proceeding:
                proceeding_docs.append(doc)
        
        # Debug information
        st.write(f"🔍 **Debug Info**:")
        st.write(f"- Total documents available: {len(self.documents)}")
        st.write(f"- Documents in proceeding '{proceeding}': {len(proceeding_docs)}")
        if len(self.documents) > 0:
            sample_proceedings = set()
            for doc in self.documents[:10]:  # Check first 10 documents
                doc_proceeding = doc.metadata.get('proceeding', '')
                source = doc.metadata.get('source', '')
                if source in self.metadata:
                    source_proceeding = self.metadata[source].get('proceeding', '')
                    if source_proceeding:
                        sample_proceedings.add(source_proceeding)
                if doc_proceeding:
                    sample_proceedings.add(doc_proceeding)
            st.write(f"- Sample proceedings found: {list(sample_proceedings)[:5]}")
        
        if not proceeding_docs:
            return {
                'answer': f"No documents found in proceeding '{proceeding}'. Available proceedings may be different.",
                'sources': [],
                'processing_stages': [],
                'cost_breakdown': self.cost_tracker
            }
        
        # Display search approach
        with st.expander("🔍 **Search Strategy**", expanded=True):
            st.write(f"**Question**: {question}")
            st.write(f"**Search Method**: Vector similarity + BM25 keyword search")
            st.write(f"**Documents in Proceeding**: {len(proceeding_docs)}")
            st.write(f"**Model Used**: {model}")
        
        # Use the existing QA system for factual queries with dynamic context
        from .qa_system import ask_question
        
        # Calculate dynamic number of results based on model and question complexity
        num_results = self._get_dynamic_num_results(model, question, len(proceeding_docs))
        
        try:
            # Get search components from session state
            vector_store = st.session_state.get('vector_store', None)
            bm25 = st.session_state.get('bm25', None)
            
            if vector_store is None or bm25 is None:
                st.warning("⚠️ Vector store or BM25 not available. Using simple text search.")
                # Fallback to simple text search with priority documents
                return self._simple_text_search_with_priority(question, proceeding_docs, model, num_results, classification)
            
            # Try to use the QA system, but with better error handling
            try:
                from .qa_system import ask_question
                
                # Use hybrid search for factual queries
                answer, sources = ask_question(
                    question=question,
                    vector_store=vector_store,
                    bm25=bm25,
                    documents=proceeding_docs,
                    metadata=self.metadata,
                    model=model,
                    search_type="Hybrid (Recommended)",
                    num_results=num_results
                )
                
                # Track costs (simplified for now)
                estimated_cost = 0.01
                self._track_cost('factual_search', estimated_cost)
                
                return {
                    'answer': answer,
                    'sources': sources,
                    'processing_stages': [{'stage': 'factual_search', 'description': f'Vector/BM25 search with {num_results} results', 'cost': estimated_cost}],
                    'cost_breakdown': self.cost_tracker
                }
                
            except Exception as qa_error:
                st.warning(f"⚠️ QA system error: {qa_error}. Falling back to simple text search.")
                # Fallback to simple text search with priority documents
                return self._simple_text_search_with_priority(question, proceeding_docs, model, num_results, classification)
            
        except Exception as e:
            st.error(f"❌ Error in factual query processing: {e}")
            # Final fallback to simple text search
            return self._simple_text_search_with_priority(question, proceeding_docs, model, num_results, classification)
    
    def _process_comparative_query(self, question: str, proceeding: str, model: str, classification: Dict) -> Dict[str, Any]:
        """Process comparative queries using vector/BM25 search with dynamic context management"""
        st.info("🔍 **Processing Comparative Query with Vector/BM25 Search**")
        
        # Filter documents by proceeding
        proceeding_docs = [doc for doc in self.documents 
                         if doc.metadata.get('proceeding', '') == proceeding]
        
        if not proceeding_docs:
            return {
                'answer': "No documents found in this proceeding.",
                'sources': [],
                'processing_stages': [],
                'cost_breakdown': self.cost_tracker
            }
        
        # Display search approach
        with st.expander("🔍 **Search Strategy**", expanded=True):
            st.write(f"**Question**: {question}")
            st.write(f"**Search Method**: Vector similarity + BM25 keyword search")
            st.write(f"**Documents in Proceeding**: {len(proceeding_docs)}")
            st.write(f"**Model Used**: {model}")
        
        # Use the existing QA system for comparative queries with dynamic context
        from .qa_system import ask_question
        
        # Calculate dynamic number of results (comparative queries need more results)
        num_results = self._get_dynamic_num_results(model, question, len(proceeding_docs))
        num_results = min(num_results * 2, 20)  # Double for comparative analysis
        
        try:
            # Get search components from session state
            vector_store = st.session_state.get('vector_store', None)
            bm25 = st.session_state.get('bm25', None)
            
            if vector_store is None or bm25 is None:
                st.warning("⚠️ Vector store or BM25 not available. Using simple text search.")
                # Fallback to simple text search
                return self._simple_text_search(question, proceeding_docs, model, num_results)
            
            # Use hybrid search for comparative queries
            answer, sources = ask_question(
                question=question,
                vector_store=vector_store,
                bm25=bm25,
                documents=proceeding_docs,
                metadata=self.metadata,
                model=model,
                search_type="Hybrid (Recommended)",
                num_results=num_results
            )
            
            # Track costs (simplified for now)
            estimated_cost = 0.015  # Slightly higher for comparative analysis
            self._track_cost('comparative_search', estimated_cost)
            
            return {
                'answer': answer,
                'sources': sources,
                'processing_stages': [{'stage': 'comparative_search', 'description': f'Vector/BM25 search with {num_results} results for comparison', 'cost': estimated_cost}],
                'cost_breakdown': self.cost_tracker
            }
            
        except Exception as e:
            st.error(f"❌ Error in comparative query processing: {e}")
            return {
                'answer': f"Error processing comparative query: {e}",
                'sources': [],
                'processing_stages': [{'stage': 'comparative_search', 'description': 'Error in search processing'}],
                'cost_breakdown': self.cost_tracker
            }
    
    def _process_general_query(self, question: str, proceeding: str, model: str, classification: Dict) -> Dict[str, Any]:
        """Process general queries using vector/BM25 search"""
        st.info("🔍 **Processing General Query with Vector/BM25 Search**")
        
        # Filter documents by proceeding
        proceeding_docs = [doc for doc in self.documents 
                         if doc.metadata.get('proceeding', '') == proceeding]
        
        if not proceeding_docs:
            return {
                'answer': "No documents found in this proceeding.",
                'sources': [],
                'processing_stages': [],
                'cost_breakdown': self.cost_tracker
            }
        
        # Display search approach
        with st.expander("🔍 **Search Strategy**", expanded=True):
            st.write(f"**Question**: {question}")
            st.write(f"**Search Method**: Vector similarity + BM25 keyword search")
            st.write(f"**Documents in Proceeding**: {len(proceeding_docs)}")
            st.write(f"**Model Used**: {model}")
        
        # Use the existing QA system for general queries with dynamic context
        from .qa_system import ask_question
        
        # Calculate dynamic number of results
        num_results = self._get_dynamic_num_results(model, question, len(proceeding_docs))
        
        try:
            # Get search components from session state
            vector_store = st.session_state.get('vector_store', None)
            bm25 = st.session_state.get('bm25', None)
            
            if vector_store is None or bm25 is None:
                st.warning("⚠️ Vector store or BM25 not available. Using simple text search.")
                # Fallback to simple text search
                return self._simple_text_search(question, proceeding_docs, model, num_results)
            
            # Use hybrid search for general queries
            answer, sources = ask_question(
                question=question,
                vector_store=vector_store,
                bm25=bm25,
                documents=proceeding_docs,
                metadata=self.metadata,
                model=model,
                search_type="Hybrid (Recommended)",
                num_results=num_results
            )
            
            # Track costs (simplified for now)
            estimated_cost = 0.01
            self._track_cost('general_search', estimated_cost)
            
            return {
                'answer': answer,
                'sources': sources,
                'processing_stages': [{'stage': 'general_search', 'description': f'Vector/BM25 search with {num_results} results', 'cost': estimated_cost}],
                'cost_breakdown': self.cost_tracker
            }
            
        except Exception as e:
            st.error(f"❌ Error in general query processing: {e}")
            return {
                'answer': f"Error processing general query: {e}",
                'sources': [],
                'processing_stages': [{'stage': 'general_search', 'description': 'Error in search processing'}],
                'cost_breakdown': self.cost_tracker
            }
