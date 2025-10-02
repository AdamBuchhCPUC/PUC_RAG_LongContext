"""
Hierarchical chunking functionality for enhanced RAG performance.
Creates multiple chunk sizes with intelligent metadata inheritance.
"""

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any, Tuple
import re
from datetime import datetime
import os


class HierarchicalChunker:
    """Creates hierarchical chunks with intelligent metadata inheritance"""
    
    def __init__(self):
        # Define chunk sizes for different levels
        self.chunk_configs = {
            'large': {
                'chunk_size': 3000,
                'chunk_overlap': 300,
                'description': 'Document sections, broad themes'
            },
            'medium': {
                'chunk_size': 1500,
                'chunk_overlap': 150,
                'description': 'Paragraphs, specific topics'
            },
            'small': {
                'chunk_size': 750,
                'chunk_overlap': 75,
                'description': 'Sentences, specific details'
            }
        }
        
        # Initialize text splitters for each level
        self.splitters = {}
        for level, config in self.chunk_configs.items():
            self.splitters[level] = RecursiveCharacterTextSplitter(
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap'],
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
    
    def create_hierarchical_chunks(self, text: str, document_metadata: Dict[str, Any]) -> List[Document]:
        """
        Create hierarchical chunks with metadata inheritance
        
        Args:
            text: Document text to chunk
            document_metadata: Rich document-level metadata
            
        Returns:
            List of Document objects with hierarchical metadata
        """
        all_chunks = []
        
        # Create chunks for each level
        for level, splitter in self.splitters.items():
            chunks = splitter.split_text(text)
            
            for i, chunk_content in enumerate(chunks):
                # Extract chunk-specific metadata
                chunk_metadata = self._extract_chunk_metadata(
                    chunk_content, 
                    document_metadata, 
                    level, 
                    i
                )
                
                # Create document with hierarchical metadata
                doc = Document(
                    page_content=chunk_content,
                    metadata=chunk_metadata
                )
                all_chunks.append(doc)
        
        # Create hierarchical relationships
        all_chunks = self._create_hierarchical_relationships(all_chunks)
        
        return all_chunks
    
    def _extract_chunk_metadata(self, chunk_content: str, document_metadata: Dict[str, Any], 
                               level: str, chunk_index: int) -> Dict[str, Any]:
        """Extract chunk-specific metadata while inheriting document metadata"""
        
        # Extract page numbers from chunk
        page_numbers = self._extract_page_numbers_from_chunk(chunk_content)
        
        # Build hierarchical metadata
        metadata = {
            # Document-level metadata (inherited)
            'source': document_metadata.get('filename', 'Unknown'),
            'proceeding': document_metadata.get('proceeding', 'Unknown'),
            'document_type': document_metadata.get('document_type', 'Unknown'),
            'filed_by': document_metadata.get('filed_by', 'Unknown'),
            'filing_date': document_metadata.get('filing_date', 'Unknown'),
            'description': document_metadata.get('description', ''),
            
            # Chunk-level metadata (specific)
            'chunk_level': level,
            'chunk_index': chunk_index,
            'chunk_size': len(chunk_content),
            'page_numbers': page_numbers,
            
            # Hierarchical relationships (will be set later)
            'parent_chunk_id': None,
            'child_chunk_ids': [],
            'sibling_chunk_ids': []
        }
        
        return metadata
    
    def _extract_page_numbers_from_chunk(self, chunk_content: str) -> List[int]:
        """Extract page numbers from chunk content"""
        page_numbers = []
        
        # Look for [PAGE X] markers
        page_pattern = r'\[PAGE (\d+)\]'
        matches = re.findall(page_pattern, chunk_content)
        
        for match in matches:
            try:
                page_numbers.append(int(match))
            except ValueError:
                continue
        
        return sorted(list(set(page_numbers)))
    
    def _create_hierarchical_relationships(self, all_chunks: List[Document]) -> List[Document]:
        """Create parent-child relationships between chunk levels"""
        
        # Group chunks by source document
        chunks_by_source = {}
        for chunk in all_chunks:
            source = chunk.metadata.get('source', 'unknown')
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
        
        # Create relationships within each document
        for source, chunks in chunks_by_source.items():
            # Sort chunks by level and position
            chunks_by_level = {
                'large': [],
                'medium': [],
                'small': []
            }
            
            for chunk in chunks:
                level = chunk.metadata.get('chunk_level', 'small')
                chunks_by_level[level].append(chunk)
            
            # Create parent-child relationships
            for i, large_chunk in enumerate(chunks_by_level['large']):
                large_chunk.metadata['chunk_id'] = f"{source}_large_{i}"
                
                # Find medium chunks that overlap with this large chunk
                for j, medium_chunk in enumerate(chunks_by_level['medium']):
                    if self._chunks_overlap(large_chunk, medium_chunk):
                        medium_chunk.metadata['parent_chunk_id'] = large_chunk.metadata['chunk_id']
                        medium_chunk.metadata['chunk_id'] = f"{source}_medium_{j}"
                        large_chunk.metadata.setdefault('child_chunk_ids', []).append(medium_chunk.metadata['chunk_id'])
                        
                        # Find small chunks that overlap with this medium chunk
                        for k, small_chunk in enumerate(chunks_by_level['small']):
                            if self._chunks_overlap(medium_chunk, small_chunk):
                                small_chunk.metadata['parent_chunk_id'] = medium_chunk.metadata['chunk_id']
                                small_chunk.metadata['chunk_id'] = f"{source}_small_{k}"
                                medium_chunk.metadata.setdefault('child_chunk_ids', []).append(small_chunk.metadata['chunk_id'])
        
        return all_chunks
    
    def _chunks_overlap(self, chunk1: Document, chunk2: Document) -> bool:
        """Check if two chunks have overlapping content"""
        content1 = chunk1.page_content.lower()
        content2 = chunk2.page_content.lower()
        
        # Simple overlap detection - if they share significant text
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        # If they share more than 30% of words, consider them overlapping
        if len(words1) > 0 and len(words2) > 0:
            overlap_ratio = len(words1.intersection(words2)) / min(len(words1), len(words2))
            return overlap_ratio > 0.3
        
        return False
