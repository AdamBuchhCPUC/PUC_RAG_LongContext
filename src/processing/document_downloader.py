"""
Document download functionality for CPUC proceedings.
Handles PDF downloads, metadata management, and file organization.
"""

import streamlit as st
import os
import json
import re
import requests
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime


def create_acronym(name: str) -> str:
    """Create an acronym from a submitter name"""
    if not name or name.strip() == '':
        return 'UNK'
    
    # Common words to skip in acronyms
    skip_words = {
        'and', 'of', 'the', 'for', 'in', 'on', 'at', 'to', 'by', 'with', 'from', 'as',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'shall', 'a', 'an', 'or', 'but', 'nor', 'so', 'yet'
    }
    
    # Split by spaces and get first letter of each significant word
    words = name.strip().split()
    acronym_letters = []
    
    for word in words:
        word_clean = word.strip('.,!?;:').lower()
        if word_clean and word_clean not in skip_words:
            acronym_letters.append(word[0].upper())
    
    # If no significant words found, use first few letters of the first word
    if not acronym_letters:
        first_word = words[0] if words else 'Unknown'
        acronym_letters = [first_word[0].upper()]
        if len(first_word) > 1:
            acronym_letters.append(first_word[1].upper())
    
    return ''.join(acronym_letters)


def get_document_suffix(document_type: str, description: str, document_index: int) -> str:
    """Generate appropriate suffix for multiple documents from same docket entry"""
    
    # Check if it's an appendix
    if 'appendix' in description.lower() or 'appendices' in description.lower():
        # Extract appendix number if possible
        appendix_match = re.search(r'appendix\s*(\d+)', description.lower())
        if appendix_match:
            return f"Appendix{appendix_match.group(1)}"
        else:
            return f"Appendix{document_index}"
    
    # For other documents, use lettering (docA, docB, docC, etc.)
    if document_index == 0:
        return "docA"
    elif document_index == 1:
        return "docB"
    elif document_index == 2:
        return "docC"
    elif document_index == 3:
        return "docD"
    elif document_index == 4:
        return "docE"
    else:
        # For more than 5 documents, use numbers
        return f"doc{document_index + 1}"


def create_document_id(metadata: Dict[str, Any]) -> str:
    """Create a unique document ID from metadata"""
    try:
        # Use proceeding, document type, and submitter to create unique ID
        proceeding = metadata.get('proceeding', 'Unknown')
        doc_type = metadata.get('document_type', 'Unknown')
        submitter = metadata.get('submitter', 'Unknown')
        
        # Create a clean ID
        clean_proceeding = re.sub(r'[^\w]', '', proceeding)
        clean_doc_type = re.sub(r'[^\w]', '', doc_type)
        clean_submitter = re.sub(r'[^\w]', '', submitter)
        
        return f"{clean_proceeding}_{clean_doc_type}_{clean_submitter}"
    except:
        return None


def download_pdfs_to_documents_folder(documents_df, documents_folder="./documents", selected_indices=None):
    """Download selected PDFs to the documents folder with metadata preservation"""
    # Create documents folder if it doesn't exist
    Path(documents_folder).mkdir(parents=True, exist_ok=True)
    
    # Create metadata folder
    metadata_folder = Path(documents_folder) / "metadata"
    metadata_folder.mkdir(exist_ok=True)
    
    if selected_indices is None:
        selected_indices = list(range(len(documents_df)))
    
    downloaded_count = 0
    failed_count = 0
    failed_downloads = []  # Track failed downloads for detailed reporting
    
    # Create progress tracking
    total_downloads = len(selected_indices)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, idx in enumerate(selected_indices):
        if idx >= len(documents_df):
            st.warning(f"⚠️ Index {idx} out of range (max: {len(documents_df)-1})")
            failed_count += 1
            continue
            
        row = documents_df.iloc[idx]
        
        # Update progress
        progress = (i + 1) / total_downloads
        progress_bar.progress(progress)
        
        try:
            # Create filename
            proceeding = row.get('proceeding', 'Unknown')
            doc_type = row.get('document_type', 'Unknown')
            submitter = row.get('submitter', 'Unknown')
            
            # Create acronym for submitter
            submitter_acronym = create_acronym(submitter)
            
            # Create base filename
            base_filename = f"{proceeding}_{doc_type}_{submitter_acronym}"
            
            # Clean filename
            clean_filename = re.sub(r'[^\w\-_]', '_', base_filename)
            clean_filename = re.sub(r'_+', '_', clean_filename)  # Remove multiple underscores
            
            # Add .pdf extension
            filename = f"{clean_filename}.pdf"
            
            # Update status
            status_text.text(f"Downloading {i+1}/{total_downloads}: {doc_type} from {submitter}")
            
            # Download PDF
            pdf_url = row.get('pdf_url', '')
            if not pdf_url:
                error_msg = f"No PDF URL available"
                st.warning(f"⚠️ Row {idx} ({doc_type}): {error_msg}")
                failed_downloads.append({
                    'index': idx,
                    'document_type': doc_type,
                    'submitter': submitter,
                    'error': error_msg
                })
                failed_count += 1
                continue
            
            # Check if file already exists
            pdf_path = Path(documents_folder) / filename
            if pdf_path.exists():
                st.info(f"⏭️ Skipping {filename} (already exists)")
                downloaded_count += 1
                continue
            
            # Download PDF with better error handling
            try:
                response = requests.get(pdf_url, timeout=30, stream=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                    st.warning(f"⚠️ Row {idx}: Unexpected content type: {content_type}")
                
                # Check file size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) < 1000:  # Less than 1KB is suspicious
                    st.warning(f"⚠️ Row {idx}: Small file size ({content_length} bytes) - may be corrupted")
                
            except requests.exceptions.Timeout:
                error_msg = "Request timeout (30s)"
                st.error(f"❌ Row {idx} ({doc_type}): {error_msg}")
                failed_downloads.append({
                    'index': idx,
                    'document_type': doc_type,
                    'submitter': submitter,
                    'error': error_msg
                })
                failed_count += 1
                continue
            except requests.exceptions.ConnectionError:
                error_msg = "Connection error"
                st.error(f"❌ Row {idx} ({doc_type}): {error_msg}")
                failed_downloads.append({
                    'index': idx,
                    'document_type': doc_type,
                    'submitter': submitter,
                    'error': error_msg
                })
                failed_count += 1
                continue
            except requests.exceptions.HTTPError as e:
                error_msg = f"HTTP error {e.response.status_code}: {e.response.reason}"
                st.error(f"❌ Row {idx} ({doc_type}): {error_msg}")
                failed_downloads.append({
                    'index': idx,
                    'document_type': doc_type,
                    'submitter': submitter,
                    'error': error_msg
                })
                failed_count += 1
                continue
            
            # Save PDF
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify file was saved and has content
            if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                error_msg = "File not saved or empty"
                st.error(f"❌ Row {idx} ({doc_type}): {error_msg}")
                failed_downloads.append({
                    'index': idx,
                    'document_type': doc_type,
                    'submitter': submitter,
                    'error': error_msg
                })
                failed_count += 1
                continue
            
            # Create metadata
            metadata = {
                'filename': filename,
                'proceeding': proceeding,
                'document_type': doc_type,
                'submitter': submitter,
                'submitter_acronym': submitter_acronym,
                'pdf_url': pdf_url,
                'download_date': datetime.now().isoformat(),
                'processed': False,
                'file_size': pdf_path.stat().st_size
            }
            
            # Save metadata
            metadata_file = metadata_folder / f"{filename}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            downloaded_count += 1
            st.success(f"✅ Downloaded: {filename} ({pdf_path.stat().st_size:,} bytes)")
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            st.error(f"❌ Row {idx} ({doc_type}): {error_msg}")
            failed_downloads.append({
                'index': idx,
                'document_type': doc_type,
                'submitter': submitter,
                'error': error_msg
            })
            failed_count += 1
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display detailed summary
    st.write(f"📊 **Download Summary**: {downloaded_count} successful, {failed_count} failed")
    
    if failed_downloads:
        with st.expander("❌ **Failed Downloads Details**", expanded=False):
            for failed in failed_downloads:
                st.write(f"**Row {failed['index']}**: {failed['document_type']} from {failed['submitter']}")
                st.write(f"  Error: {failed['error']}")
                st.write("---")
    
    return downloaded_count, failed_count


def load_documents_metadata(documents_folder="./documents"):
    """Load metadata for documents in the folder"""
    metadata_folder = Path(documents_folder) / "metadata"
    
    if not metadata_folder.exists():
        return {}
    
    metadata_dict = {}
    
    for metadata_file in metadata_folder.glob("*.json"):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                filename = metadata.get('filename', metadata_file.stem)
                metadata_dict[filename] = metadata
        except Exception as e:
            st.warning(f"Could not load metadata from {metadata_file}: {e}")
    
    return metadata_dict


def check_if_processing_needed(documents_folder="./documents"):
    """Check if documents need processing (conversion to text)"""
    metadata_dict = load_documents_metadata(documents_folder)
    
    unprocessed_count = 0
    processed_count = 0
    
    for filename, metadata in metadata_dict.items():
        if metadata.get('processed', False):
            processed_count += 1
        else:
            unprocessed_count += 1
    
    return {
        'total_documents': len(metadata_dict),
        'processed': processed_count,
        'unprocessed': unprocessed_count,
        'needs_processing': unprocessed_count > 0
    }
