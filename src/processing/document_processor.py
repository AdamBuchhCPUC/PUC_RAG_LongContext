"""
Simplified document processing functionality for CPUC proceedings.
Handles PDF text extraction, hierarchical chunking, and vector store creation.
"""

import streamlit as st
import os
import pickle
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
import json
from datetime import datetime

# PDF processing
import PyPDF2

# Text processing and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# BM25 for keyword search
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import hierarchical chunker and cache manager
from .hierarchical_chunker import HierarchicalChunker
from .cache_manager import ContentHashCache


class DocumentProcessor:
    """Simplified document processor for CPUC documents"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize document processor"""
        self.model = model
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model, temperature=0.1, request_timeout=120)
        
        # Initialize hierarchical chunker
        self.hierarchical_chunker = HierarchicalChunker()
        
        # Initialize cache manager
        self.cache_manager = ContentHashCache()
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += f"[PAGE {page_num + 1}]\n{page_text}\n\n"
                
                return text
        except Exception as e:
            st.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_documents(self, documents_folder: str) -> Tuple[Any, Any, List[Document], Dict]:
        """Process documents using hierarchical chunking"""
        documents = []
        metadata = {}
        processed_count = 0
        
        # Load existing metadata
        from .document_downloader import load_documents_metadata
        existing_metadata = load_documents_metadata(documents_folder)
        
        st.write(f"🔍 Loaded metadata for {len(existing_metadata)} documents")
        
        # Process each PDF
        pdf_files = []
        # Look for PDFs in the main documents folder
        pdf_files.extend(list(Path(documents_folder).glob("*.pdf")))
        # Look for PDFs in proceeding subfolders
        for proceeding_folder in Path(documents_folder).iterdir():
            if proceeding_folder.is_dir() and proceeding_folder.name != "metadata":
                pdf_files.extend(list(proceeding_folder.glob("*.pdf")))
        
        st.write(f"📄 Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files):
            try:
                st.write(f"📄 Processing {i+1}/{len(pdf_files)}: {pdf_path.name}")
                
                # Extract text
                text = self.extract_text_from_pdf(str(pdf_path))
                if not text.strip():
                    st.warning(f"⚠️ No text extracted from {pdf_path.name}")
                    continue
                
                # Get document metadata
                doc_metadata = existing_metadata.get(pdf_path.name, {})
                doc_metadata['filename'] = pdf_path.name
                doc_metadata['processed'] = True
                
                # Create hierarchical chunks
                chunks = self.hierarchical_chunker.create_hierarchical_chunks(text, doc_metadata)
                documents.extend(chunks)
                
                # Store metadata
                metadata[pdf_path.name] = doc_metadata
                processed_count += 1
                
                st.success(f"✅ Processed {pdf_path.name} - {len(chunks)} chunks created")
                
            except Exception as e:
                st.error(f"❌ Error processing {pdf_path.name}: {e}")
                continue
        
        if not documents:
            st.error("❌ No documents were processed successfully")
            return None, None, [], {}
        
        st.write(f"🔄 Creating vector store and BM25 index...")
        
        # Create vector store with caching
        vector_store = self._create_vector_store_with_cache(documents)
        
        # Create BM25 index
        bm25 = self._create_bm25_index(documents)
        
        st.success(f"✅ Processing complete! {processed_count} documents, {len(documents)} chunks")
        
        return vector_store, bm25, documents, metadata
    
    def _create_vector_store_with_cache(self, documents: List[Document]) -> Any:
        """Create vector store with content-hash caching"""
        try:
            # Check cache for existing embeddings
            cached_embeddings = []
            new_documents = []
            
            for doc in documents:
                content = doc.page_content
                is_cached, embedding, cached_metadata = self.cache_manager.get_cached_embedding(content)
                
                if is_cached:
                    cached_embeddings.append(embedding)
                else:
                    new_documents.append(doc)
            
            # Create embeddings for new documents
            if new_documents:
                st.write(f"🔄 Creating embeddings for {len(new_documents)} new chunks...")
                new_embeddings = self.embeddings.embed_documents([doc.page_content for doc in new_documents])
                
                # Cache new embeddings
                for doc, embedding in zip(new_documents, new_embeddings):
                    self.cache_manager.cache_embedding(doc.page_content, embedding, doc.metadata)
                
                # Combine cached and new embeddings
                all_embeddings = cached_embeddings + new_embeddings
                all_documents = documents
            else:
                st.write("✅ All embeddings found in cache!")
                all_embeddings = cached_embeddings
                all_documents = documents
            
            # Create FAISS vector store
            vector_store = FAISS.from_embeddings(
                text_embeddings=list(zip([doc.page_content for doc in all_documents], all_embeddings)),
                embedding=self.embeddings,
                metadatas=[doc.metadata for doc in all_documents]
            )
            
            return vector_store
            
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None
    
    def _create_bm25_index(self, documents: List[Document]) -> Any:
        """Create BM25 keyword search index"""
        try:
            # Tokenize documents for BM25
            tokenized_docs = []
            for doc in documents:
                tokens = word_tokenize(doc.page_content.lower())
                tokenized_docs.append(tokens)
            
            # Create BM25 index
            bm25 = BM25Okapi(tokenized_docs)
            
            return bm25
            
        except Exception as e:
            st.error(f"Error creating BM25 index: {e}")
            return None
    
    def save_processed_data(self, vector_store, bm25, documents, metadata):
        """Save processed data to disk"""
        try:
            # Save vector store
            vector_store.save_local("./processed_data_vector")
            
            # Save BM25 index
            with open("./processed_data_vector/processed_data_bm25.pkl", "wb") as f:
                pickle.dump(bm25, f)
            
            # Save documents and metadata
            with open("./processed_data_vector/processed_data_documents.pkl", "wb") as f:
                pickle.dump(documents, f)
            
            with open("./processed_data_vector/processed_data_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            st.success("✅ Processed data saved successfully!")
            
        except Exception as e:
            st.error(f"Error saving processed data: {e}")
    
    def load_processed_data(self, data_path: str = "./processed_data"):
        """Load processed data from disk"""
        try:
            # Load vector store
            vector_store = FAISS.load_local("./processed_data_vector", self.embeddings, allow_dangerous_deserialization=True)
            
            # Load BM25 index
            with open("./processed_data_vector/processed_data_bm25.pkl", "rb") as f:
                bm25 = pickle.load(f)
            
            # Load documents and metadata
            with open("./processed_data_vector/processed_data_documents.pkl", "rb") as f:
                documents = pickle.load(f)
            
            with open("./processed_data_vector/processed_data_metadata.json", "r") as f:
                metadata = json.load(f)
            
            return vector_store, bm25, documents, metadata
            
        except Exception as e:
            st.error(f"Error loading processed data: {e}")
            return None, None, [], {}
    
    def clear_processed_data(self):
        """Clear all processed data to start fresh"""
        try:
            # Remove processed data files
            files_to_remove = [
                "./processed_data_vector",
                "./processed_data_vector/processed_data_bm25.pkl", 
                "./processed_data_vector/processed_data_documents.pkl",
                "./processed_data_vector/processed_data_metadata.json"
            ]
            
            cleared_count = 0
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    try:
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        else:
                            os.remove(file_path)
                        cleared_count += 1
                    except Exception as e:
                        st.warning(f"Could not remove {file_path}: {e}")
            
            if cleared_count > 0:
                st.success(f"✅ Cleared {cleared_count} processed data files!")
                return True
            else:
                st.warning("⚠️ No files could be cleared - they may be in use")
                return False
                
        except Exception as e:
            st.error(f"Error clearing processed data: {e}")
            return False
