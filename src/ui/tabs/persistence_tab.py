"""
Persistence Tab
Handles manual download/upload of processed data for persistence across sessions.
"""

import streamlit as st
import zipfile
import json
import pickle
import io
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import os


def create_persistence_tab():
    """Create the data persistence tab"""
    st.header("💾 Data Persistence")
    st.markdown("Download your processed data to save it, or upload previously saved data to restore it.")
    
    # Check if data is available for download
    vector_available = st.session_state.get('vector_store') is not None
    documents_available = st.session_state.get('documents') is not None
    metadata_available = st.session_state.get('metadata') is not None
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Download Processed Data")
        
        if vector_available and documents_available and metadata_available:
            st.success("✅ Data is ready for download!")
            
            # Show data summary
            st.write(f"**Documents**: {len(st.session_state.documents)} chunks")
            st.write(f"**Metadata**: {len(st.session_state.metadata)} files")
            st.write(f"**Vector Store**: Available")
            st.write(f"**BM25 Index**: Available")
            
            if st.button("📦 Download All Data", type="primary"):
                download_processed_data()
        else:
            st.warning("⚠️ No processed data available. Process some documents first.")
    
    with col2:
        st.subheader("📤 Upload Saved Data")
        
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['zip'],
            help="Upload a previously downloaded data file to restore your processed documents"
        )
        
        if uploaded_file is not None:
            st.info("📁 File uploaded successfully!")
            st.write(f"**File name**: {uploaded_file.name}")
            st.write(f"**File size**: {uploaded_file.size:,} bytes")
            
            if st.button("🔄 Restore Data", type="primary"):
                restore_processed_data(uploaded_file)
    
    # Instructions
    st.subheader("📋 How to Use Data Persistence")
    
    st.markdown("""
    **To Save Your Work:**
    1. Process your documents using the "Process Documents" tab
    2. Come to this tab and click "Download All Data"
    3. Save the downloaded ZIP file to your computer
    
    **To Restore Your Work:**
    1. Upload the ZIP file you previously downloaded
    2. Click "Restore Data"
    3. Your processed documents will be available in the "Ask Questions" tab
    
    **What Gets Saved:**
    - ✅ All document chunks and embeddings
    - ✅ Vector search index
    - ✅ BM25 keyword search index
    - ✅ Document metadata
    - ✅ Processing settings and timestamps
    
    **Benefits:**
    - 🔄 **Persistent**: Your work survives between sessions
    - 🚀 **Fast**: No need to re-process documents
    - 💾 **Portable**: Share data files with colleagues
    - 🔒 **Private**: Your data stays on your computer
    """)


def download_processed_data():
    """Download all processed data as a ZIP file"""
    try:
        with st.spinner("📦 Creating data package..."):
            # Create a temporary ZIP file in memory
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Save vector store
                if st.session_state.vector_store:
                    # Create a temporary directory for vector store
                    with tempfile.TemporaryDirectory() as temp_dir:
                        vector_path = Path(temp_dir) / "vector_store"
                        st.session_state.vector_store.save_local(str(vector_path))
                        
                        # Add vector store files to ZIP
                        for file_path in vector_path.rglob("*"):
                            if file_path.is_file():
                                arcname = file_path.relative_to(vector_path)
                                zip_file.write(file_path, f"vector_store/{arcname}")
                
                # Save BM25 index
                if st.session_state.bm25:
                    bm25_data = pickle.dumps(st.session_state.bm25)
                    zip_file.writestr("bm25_index.pkl", bm25_data)
                
                # Save documents
                if st.session_state.documents:
                    documents_data = pickle.dumps(st.session_state.documents)
                    zip_file.writestr("documents.pkl", documents_data)
                
                # Save metadata
                if st.session_state.metadata:
                    metadata_json = json.dumps(st.session_state.metadata, indent=2)
                    zip_file.writestr("metadata.json", metadata_json)
                
                # Save session info
                session_info = {
                    "total_qa_costs": st.session_state.get('total_qa_costs', 0),
                    "download_timestamp": st.datetime.now().isoformat(),
                    "document_count": len(st.session_state.documents) if st.session_state.documents else 0,
                    "metadata_count": len(st.session_state.metadata) if st.session_state.metadata else 0
                }
                zip_file.writestr("session_info.json", json.dumps(session_info, indent=2))
            
            # Prepare download
            zip_buffer.seek(0)
            
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"puc_rag_data_{timestamp}.zip"
            
            st.download_button(
                label="💾 Download Data File",
                data=zip_buffer.getvalue(),
                file_name=filename,
                mime="application/zip",
                help="Save this file to restore your processed data later"
            )
            
            st.success("✅ Data package created successfully!")
            st.info(f"📁 **File**: {filename}")
            st.info(f"📊 **Contents**: {len(st.session_state.documents)} chunks, {len(st.session_state.metadata)} metadata entries")
            
    except Exception as e:
        st.error(f"❌ Error creating data package: {e}")


def restore_processed_data(uploaded_file):
    """Restore processed data from uploaded ZIP file"""
    try:
        with st.spinner("🔄 Restoring data..."):
            # Read the uploaded file
            zip_data = uploaded_file.read()
            
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_file:
                # Check if required files exist
                required_files = ["documents.pkl", "metadata.json", "bm25_index.pkl"]
                missing_files = [f for f in required_files if f not in zip_file.namelist()]
                
                if missing_files:
                    st.error(f"❌ Missing required files: {missing_files}")
                    return
                
                # Restore documents
                if "documents.pkl" in zip_file.namelist():
                    documents_data = zip_file.read("documents.pkl")
                    st.session_state.documents = pickle.loads(documents_data)
                    st.success(f"✅ Restored {len(st.session_state.documents)} document chunks")
                
                # Restore metadata
                if "metadata.json" in zip_file.namelist():
                    metadata_json = zip_file.read("metadata.json").decode('utf-8')
                    st.session_state.metadata = json.loads(metadata_json)
                    st.success(f"✅ Restored {len(st.session_state.metadata)} metadata entries")
                
                # Restore BM25 index
                if "bm25_index.pkl" in zip_file.namelist():
                    bm25_data = zip_file.read("bm25_index.pkl")
                    st.session_state.bm25 = pickle.loads(bm25_data)
                    st.success("✅ Restored BM25 keyword search index")
                
                # Restore vector store
                if any(name.startswith("vector_store/") for name in zip_file.namelist()):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Extract vector store files
                        for file_info in zip_file.filelist:
                            if file_info.filename.startswith("vector_store/"):
                                file_info.filename = file_info.filename[13:]  # Remove "vector_store/" prefix
                                zip_file.extract(file_info, temp_dir)
                        
                        # Load vector store
                        from processing.document_processor import DocumentProcessor
                        processor = DocumentProcessor()
                        vector_store = processor._load_vector_store_from_path(Path(temp_dir) / "vector_store")
                        
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.success("✅ Restored vector search index")
                
                # Restore session info
                if "session_info.json" in zip_file.namelist():
                    session_info_json = zip_file.read("session_info.json").decode('utf-8')
                    session_info = json.loads(session_info_json)
                    
                    if 'total_qa_costs' in session_info:
                        st.session_state.total_qa_costs = session_info['total_qa_costs']
                    
                    st.info(f"📊 **Original data**: {session_info.get('document_count', 0)} chunks, {session_info.get('metadata_count', 0)} metadata entries")
                    st.info(f"📅 **Saved on**: {session_info.get('download_timestamp', 'Unknown')}")
            
            st.success("🎉 Data restored successfully!")
            st.info("💡 You can now use the 'Ask Questions' tab to search through your restored documents.")
            
            # Force rerun to update the interface
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error restoring data: {e}")
        st.error("Make sure you're uploading a valid data file created by this application.")


def get_data_summary() -> Dict[str, Any]:
    """Get summary of current data state"""
    return {
        "documents_count": len(st.session_state.documents) if st.session_state.documents else 0,
        "metadata_count": len(st.session_state.metadata) if st.session_state.metadata else 0,
        "vector_store_available": st.session_state.vector_store is not None,
        "bm25_available": st.session_state.bm25 is not None,
        "total_qa_costs": st.session_state.get('total_qa_costs', 0)
    }
