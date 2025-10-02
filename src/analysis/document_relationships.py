"""
Document relationship analysis for CPUC proceedings.
Handles response detection, party relationships, and chronological ordering.
"""

import streamlit as st
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from datetime import datetime
import re


class DocumentRelationshipAnalyzer:
    """Analyzes relationships between documents in CPUC proceedings"""
    
    def __init__(self, documents: List[Document], metadata: Dict[str, Any]):
        """Initialize document relationship analyzer"""
        self.documents = documents
        self.metadata = metadata
        
        # Relationship tracking
        self.relationship_stats = {
            'response_documents_found': 0,
            'party_relationships_found': 0,
            'chronological_relationships_found': 0,
            'total_relationships_analyzed': 0
        }
    
    def find_responses_to_document(self, selected_document: Dict[str, Any]) -> List[Document]:
        """Find documents that are responses to the selected document"""
        if not selected_document:
            return []
        
        # Removed verbose output
        
        response_docs = []
        target_doc_type = selected_document.get('document_type', '').lower()
        target_filed_by = selected_document.get('filed_by', '').lower()
        target_description = selected_document.get('description', '').lower()
        target_filename = selected_document.get('filename', '')
        
        # Keywords that indicate responses
        response_keywords = [
            'response', 'reply', 'comment', 'protest', 'opposition', 'support', 
            'objection', 'agreement', 'disagreement', 'concern', 'recommendation',
            'suggestion', 'alternative', 'modification', 'amendment', 'rebuttal',
            'supplement', 'clarification', 'correction'
        ]
        
        # Keywords that indicate what they're responding to
        target_keywords = [
            'motion', 'application', 'decision', 'proposed decision', 'ruling', 
            'order', 'proposal', 'request', 'petition', 'application', 'complaint'
        ]
        
        # Extract key terms from target document for better matching
        target_terms = self._extract_key_terms(target_description)
        
        for doc in self.documents:
            doc_metadata = doc.metadata
            source = doc_metadata.get('source', '')
            
            # Skip if this is the same document
            if source == target_filename:
                continue
            
            # Get document metadata
            if source in self.metadata:
                doc_meta = self.metadata[source]
                doc_type = doc_meta.get('document_type', '').lower()
                doc_filed_by = doc_meta.get('filed_by', '').lower()
                doc_description = doc_meta.get('description', '').lower()
                doc_date = doc_meta.get('filing_date', '')
            else:
                doc_type = doc_metadata.get('document_type', '').lower()
                doc_filed_by = doc_metadata.get('filed_by', '').lower()
                doc_description = doc_metadata.get('description', '').lower()
                doc_date = doc_metadata.get('filing_date', '')
            
            # Calculate response score
            response_score = self._calculate_response_score(
                doc_type, doc_filed_by, doc_description, doc_date,
                target_doc_type, target_filed_by, target_description,
                response_keywords, target_keywords, target_terms
            )
            
            if response_score >= 0.3:  # Threshold for response relationship
                response_docs.append({
                    'document': doc,
                    'score': response_score,
                    'relationship_type': 'response',
                    'reason': self._get_response_reason(
                        doc_type, doc_description, target_doc_type, response_score
                    )
                })
        
        # Sort by response score
        response_docs.sort(key=lambda x: x['score'], reverse=True)
        
        self.relationship_stats['response_documents_found'] = len(response_docs)
        self.relationship_stats['total_relationships_analyzed'] += 1
        
        # Removed verbose output
        
        return response_docs
    
    def find_party_relationships(self, selected_document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find documents from the same party in the same proceeding (simplified)"""
        if not selected_document:
            return []
        
        target_filed_by = selected_document.get('filed_by', '').lower()
        target_proceeding = selected_document.get('proceeding', '')
        
        # Removed verbose output
        
        party_relationships = []
        
        # Find documents from the same party in the same proceeding
        for doc in self.documents:
            source = doc.metadata.get('source', '')
            
            if source in self.metadata:
                doc_meta = self.metadata[source]
                doc_filed_by = doc_meta.get('filed_by', '').lower()
                doc_proceeding = doc_meta.get('proceeding', '')
            else:
                doc_filed_by = doc.metadata.get('filed_by', '').lower()
                doc_proceeding = doc.metadata.get('proceeding', '')
            
            # Check if from same party and same proceeding
            if (doc_proceeding == target_proceeding and 
                doc_filed_by == target_filed_by):
                
                party_relationships.append({
                    'document': doc,
                    'score': 0.8,  # High score for same party
                    'relationship_type': 'same_party',
                    'reason': f"Same party: {doc_filed_by}"
                })
        
        # Sort by score
        party_relationships.sort(key=lambda x: x['score'], reverse=True)
        
        self.relationship_stats['party_relationships_found'] = len(party_relationships)
        
        # Removed verbose output
        
        return party_relationships
    
    def find_chronological_relationships(self, selected_document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find documents in chronological relationship"""
        if not selected_document:
            return []
        
        target_date = selected_document.get('filing_date', '')
        target_proceeding = selected_document.get('proceeding', '')
        
        st.info(f"🔍 Finding chronological relationships for: {target_date}")
        
        chronological_relationships = []
        
        for doc in self.documents:
            source = doc.metadata.get('source', '')
            
            if source in self.metadata:
                doc_meta = self.metadata[source]
                doc_date = doc_meta.get('filing_date', '')
                doc_proceeding = doc_meta.get('proceeding', '')
            else:
                doc_date = doc.metadata.get('filing_date', '')
                doc_proceeding = doc.metadata.get('proceeding', '')
            
            # Check if same proceeding
            if doc_proceeding == target_proceeding:
                # Calculate chronological relationship
                time_relationship = self._calculate_time_relationship(target_date, doc_date)
                
                if time_relationship['relationship'] != 'same':
                    chronological_relationships.append({
                        'document': doc,
                        'score': time_relationship['score'],
                        'relationship_type': 'chronological',
                        'reason': f"{time_relationship['relationship']} document: {doc_date} vs {target_date}",
                        'time_difference': time_relationship['days_difference']
                    })
        
        # Sort by time difference (closer documents first)
        chronological_relationships.sort(key=lambda x: abs(x['time_difference']))
        
        self.relationship_stats['chronological_relationships_found'] = len(chronological_relationships)
        
        st.success(f"✅ Found {len(chronological_relationships)} chronological relationship documents")
        
        return chronological_relationships
    
    def find_responses_to_specific_document(self, target_document: Dict[str, Any], 
                                          exclude_other_originating_docs: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find responses specifically to a target document, excluding responses to other originating documents"""
        if not target_document:
            return []
        
        st.info(f"🔍 Finding responses specifically to: {target_document.get('document_type', 'Unknown')}")
        
        # Get all potential response documents
        all_responses = self.find_responses_to_document(target_document)
        
        if not exclude_other_originating_docs:
            return all_responses
        
        # Filter out responses that are more likely to be responding to other documents
        filtered_responses = []
        
        for response in all_responses:
            doc = response['document']
            doc_meta = doc.metadata
            
            # Check if this response is more likely to be responding to other originating documents
            is_more_likely_other = False
            
            for other_doc in exclude_other_originating_docs:
                other_score = self._calculate_response_score(
                    doc_meta.get('document_type', '').lower(),
                    doc_meta.get('filed_by', '').lower(),
                    doc_meta.get('description', '').lower(),
                    doc_meta.get('filing_date', ''),
                    other_doc.get('document_type', '').lower(),
                    other_doc.get('filed_by', '').lower(),
                    other_doc.get('description', '').lower(),
                    ['response', 'reply', 'comment', 'protest', 'objection', 'support', 'opposition'],
                    ['motion', 'application', 'decision', 'proposed decision', 'ruling', 'order', 'proposal'],
                    self._extract_key_terms(other_doc.get('description', ''))
                )
                
                # If this response scores higher for another document, it's more likely responding to that one
                if other_score > response['score']:
                    is_more_likely_other = True
                    break
            
            if not is_more_likely_other:
                filtered_responses.append(response)
        
        st.success(f"✅ Found {len(filtered_responses)} responses specifically to target document")
        return filtered_responses
    
    def _calculate_response_score(self, doc_type: str, doc_filed_by: str, doc_description: str, 
                                doc_date: str, target_doc_type: str, target_filed_by: str, 
                                target_description: str, response_keywords: List[str], 
                                target_keywords: List[str], target_terms: List[str]) -> float:
        """Calculate response relationship score with enhanced document type matching"""
        score = 0.0
        
        # Check if this document is a response type
        is_response = any(keyword in doc_type for keyword in response_keywords)
        if is_response:
            score += 0.3
        
        # Enhanced document type matching for specific CPUC patterns
        doc_type_score = self._calculate_document_type_response_score(doc_type, target_doc_type)
        score += doc_type_score
        
        # Check for direct mentions of target document elements
        if target_doc_type in doc_description:
            score += 0.2
        if target_filed_by in doc_description:
            score += 0.2
        
        # Check for mentions of target document type
        if any(keyword in doc_description for keyword in target_keywords if keyword in target_description):
            score += 0.2
        
        # Check for key terms from target document
        doc_terms = self._extract_key_terms(doc_description)
        common_terms = set(target_terms) & set(doc_terms)
        if common_terms:
            score += len(common_terms) * 0.05
        
        # Check for chronological relationship (documents filed after target)
        if self._is_document_after_target(doc_date, target_description):
            score += 0.1
        
        # Check for party relationships (different parties responding)
        if doc_filed_by != target_filed_by and is_response:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_document_type_response_score(self, doc_type: str, target_doc_type: str) -> float:
        """Calculate score based on document type relationships in CPUC proceedings"""
        score = 0.0
        
        # Define response patterns for CPUC proceedings
        response_patterns = {
            # Responses to motions
            'motion': ['response', 'reply', 'comment', 'protest', 'objection', 'support', 'opposition'],
            # Responses to proposed decisions
            'proposed decision': ['response', 'reply', 'comment', 'protest', 'objection', 'support', 'opposition', 'brief', 'comment'],
            # Responses to scoping rulings
            'scoping ruling': ['response', 'reply', 'comment', 'brief', 'comment'],
            # Responses to applications
            'application': ['response', 'reply', 'comment', 'protest', 'objection', 'support', 'opposition'],
            # Responses to petitions
            'petition': ['response', 'reply', 'comment', 'protest', 'objection', 'support', 'opposition']
        }
        
        # Check if the document type suggests a response to the target
        target_lower = target_doc_type.lower()
        doc_lower = doc_type.lower()
        
        # Look for specific response patterns
        for target_pattern, response_types in response_patterns.items():
            if target_pattern in target_lower:
                for response_type in response_types:
                    if response_type in doc_lower:
                        score += 0.4  # Strong indicator of response relationship
                        break
        
        # Additional scoring for common CPUC response patterns
        if 'brief' in doc_lower and ('decision' in target_lower or 'ruling' in target_lower):
            score += 0.3
        if 'comment' in doc_lower and ('motion' in target_lower or 'application' in target_lower):
            score += 0.3
        if 'reply' in doc_lower and ('motion' in target_lower or 'application' in target_lower):
            score += 0.3
        
        return min(score, 0.4)  # Cap at 0.4 to leave room for other factors
    
    
    def _calculate_time_relationship(self, target_date: str, doc_date: str) -> Dict[str, Any]:
        """Calculate chronological relationship between documents"""
        try:
            # Parse dates (assuming format like "August 28, 2025" or "08/28/2025")
            target_dt = self._parse_date(target_date)
            doc_dt = self._parse_date(doc_date)
            
            if not target_dt or not doc_dt:
                return {'relationship': 'unknown', 'score': 0.0, 'days_difference': 0}
            
            days_diff = (doc_dt - target_dt).days
            
            if days_diff == 0:
                return {'relationship': 'same', 'score': 1.0, 'days_difference': 0}
            elif days_diff > 0:
                return {'relationship': 'after', 'score': 0.8, 'days_difference': days_diff}
            else:
                return {'relationship': 'before', 'score': 0.6, 'days_difference': abs(days_diff)}
                
        except Exception:
            return {'relationship': 'unknown', 'score': 0.0, 'days_difference': 0}
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'
        }
        
        # Extract words longer than 3 characters
        words = re.findall(r'\b\w{4,}\b', text.lower())
        key_terms = [word for word in words if word not in stop_words]
        
        return key_terms
    
    def _is_document_after_target(self, doc_date: str, target_date: str) -> bool:
        """Check if document was filed after target document"""
        try:
            doc_dt = self._parse_date(doc_date)
            target_dt = self._parse_date(target_date)
            
            if doc_dt and target_dt:
                return doc_dt > target_dt
        except:
            pass
        
        return False
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string into datetime object"""
        if not date_str:
            return None
        
        # Try different date formats
        date_formats = [
            "%B %d, %Y",      # August 28, 2025
            "%B %d,%Y",       # August 28,2025
            "%m/%d/%Y",       # 08/28/2025
            "%m-%d-%Y",       # 08-28-2025
            "%Y-%m-%d"        # 2025-08-28
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        return None
    
    def _get_response_reason(self, doc_type: str, doc_description: str, 
                           target_doc_type: str, score: float) -> str:
        """Get human-readable reason for response relationship"""
        if score >= 0.7:
            return f"Strong response relationship (score: {score:.2f})"
        elif score >= 0.5:
            return f"Moderate response relationship (score: {score:.2f})"
        else:
            return f"Weak response relationship (score: {score:.2f})"
    
    def get_relationship_stats(self) -> Dict[str, Any]:
        """Get relationship analysis statistics"""
        return self.relationship_stats.copy()
    
    def reset_stats(self):
        """Reset relationship statistics"""
        self.relationship_stats = {
            'response_documents_found': 0,
            'party_relationships_found': 0,
            'chronological_relationships_found': 0,
            'total_relationships_analyzed': 0
        }
