"""
Download Tab
Handles document downloading functionality with sophisticated proceeding-based download.
"""

import streamlit as st
from pathlib import Path
from processing.document_downloader import load_documents_metadata, check_if_processing_needed
from processing.document_scraper import DocumentCache, CPUCSeleniumScraper


def create_download_tab():
    """Create the document download tab"""
    st.header("📥 Download CPUC Documents")
    st.markdown("Download PDFs from CPUC proceedings with intelligent filtering and time-based selection")
    
    # Show current status
    documents_folder = "./documents"
    if Path(documents_folder).exists():
        processing_status = check_if_processing_needed(documents_folder)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", processing_status['total_documents'])
        with col2:
            st.metric("Processed", processing_status['processed'])
        with col3:
            st.metric("Unprocessed", processing_status['unprocessed'])
        
        if processing_status['needs_processing']:
            st.info("📄 Some documents need processing. Use the 'Process Documents' tab to convert them to searchable text.")
        else:
            st.success("✅ All documents are processed and ready for Q&A!")
    else:
        st.info("📁 No documents folder found. Download some documents to get started.")
    
    # Show cache status and management
    cache = DocumentCache()
    downloads_cache = cache.get_downloads_cache()
    if downloads_cache:
        st.info(f"📋 Cache contains {len(downloads_cache)} proceeding download records")
        
        # Show cache entries
        with st.expander("📋 View Cache Entries"):
            for cache_key, cache_data in downloads_cache.items():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**Proceeding {cache_data.get('proceeding', 'Unknown')}** - {cache_data.get('documents_count', 0)} documents")
                with col2:
                    st.write(f"Downloaded: {cache_data.get('download_time', 'Unknown')[:19]}")
                with col3:
                    if st.button("🗑️ Clear", key=f"clear_{cache_key}"):
                        # Clear specific cache entry
                        del downloads_cache[cache_key]
                        cache.save_downloads_cache(downloads_cache)
                        st.success("Cache entry cleared!")
                        st.rerun()
    
    # Proceeding input
    col1, col2 = st.columns([2, 1])
    with col1:
        proceeding_number = st.text_input(
            "Enter CPUC Proceeding Number:",
            placeholder="R2008020",
            help="Enter proceeding number (e.g., R2008020, A2106028)"
        )
    
    with col2:
        max_pages = st.number_input(
            "Max pages:",
            min_value=1,
            max_value=50,
            value=10,
            help="Limit pages to scrape"
        )
    
    # Advanced filtering options
    with st.expander("Document Filtering Options", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Time Period Filter")
            time_filter = st.selectbox(
                "Select time period:",
                ["Whole docket", "Last 30 days", "Last 60 days", "Last 90 days", "Last 180 days", 
                 "Last 12 months", "Last 18 months", "Last 24 months", "Last 36 months",
                 "Since 2020", "Since 2019", "Since 2018", "Since 2017", "Since 2016", 
                 "Since 2015", "Since 2014", "Since 2013", "Since 2012", "Since 2011", "Since 2010"],
                help="Limit documents to recent time periods"
            )
        
        with col2:
            st.subheader("Stop at Key Document")
            keyword_filter = st.selectbox(
                "Stop scraping when finding:",
                ["None", "PROPOSED DECISION", "SCOPING RULING", "SCOPING MEMO", "DECISION", "RULING"],
                help="Stop when reaching the last occurrence of this document type"
            )
        
        with col3:
            st.subheader("Browser Options")
            headless = st.checkbox(
                "Run browser in background",
                value=True,
                help="Uncheck to see browser window (useful for debugging)"
            )
    
    # Additional filtering options
    with st.expander("Advanced Document Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Document Type Filters")
            filter_intervenor_comp = st.checkbox(
                "Skip intervenor compensation documents",
                value=True,  # Enabled by default as requested
                help="Filter out documents with 'intervenor compensation' in description"
            )
        
        with col2:
            st.subheader("Content Filters")
            filter_short_docs = st.checkbox(
                "Skip very short documents (< 5 pages)",
                value=False,
                help="Filter out documents that are likely too short to be useful"
            )
    
    # Download button
    if st.button("🚀 Download Documents", type="primary"):
        if proceeding_number:
            try:
                # Initialize scraper
                scraper = CPUCSeleniumScraper(headless=headless)
                
                # Show progress
                with st.spinner(f"🔍 Scraping proceeding {proceeding_number}..."):
                    downloaded_count = scraper.scrape_proceeding(
                        proceeding_number=proceeding_number,
                        time_filter=time_filter,
                        keyword_filter=keyword_filter,
                        max_pages=max_pages
                    )
                
                if downloaded_count > 0:
                    st.success(f"✅ Successfully downloaded {downloaded_count} documents!")
                    st.info("💡 Use the 'Process Documents' tab to convert them to searchable text.")
                    st.rerun()
                else:
                    st.warning("⚠️ No documents were downloaded. Check the proceeding number and try again.")
                    
            except Exception as e:
                st.error(f"❌ Error downloading documents: {e}")
                st.info("💡 Make sure Chrome and ChromeDriver are installed for web scraping.")
        else:
            st.error("Please enter a proceeding number")
    
    # Instructions
    st.subheader("📋 Instructions")
    st.markdown("""
    **How to download documents:**
    
    1. **Enter Proceeding Number**: Use the format like R2008020, A2106028
    2. **Set Time Filter**: Choose how far back to look for documents
    3. **Set Stop Condition**: Choose when to stop downloading (e.g., at last decision)
    4. **Download**: Click the download button to start the process
    5. **Process Documents**: Use the 'Process Documents' tab to convert PDFs to searchable text
    
    **Advanced Features:**
    - **Time-based filtering**: Download only recent documents
    - **Smart stopping**: Stop at key documents like decisions or rulings
    - **Caching**: Avoid re-downloading the same proceeding
    - **Filtering**: Skip unwanted document types
    
    **Folder Structure:**
    ```
    ./documents/
    ├── R2008020/           # Proceeding subfolder
    │   ├── R2008020_001.pdf
    │   ├── R2008020_002.pdf
    │   └── ...
    └── cache/              # Download cache
        └── downloads_cache.json
    ```
    """)
    
    # Show existing documents
    if Path(documents_folder).exists():
        st.subheader("📁 Current Documents")
        
        # Load and display metadata
        metadata = load_documents_metadata(documents_folder)
        
        if metadata:
            st.write(f"Found {len(metadata)} documents:")
            
            for filename, doc_metadata in list(metadata.items())[:10]:  # Show first 10
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 **{filename}**")
                    if 'proceeding' in doc_metadata:
                        st.write(f"   Proceeding: {doc_metadata['proceeding']}")
                    if 'document_type' in doc_metadata:
                        st.write(f"   Type: {doc_metadata['document_type']}")
                
                with col2:
                    if doc_metadata.get('processed', False):
                        st.success("✅ Processed")
                    else:
                        st.warning("⏳ Pending")
                
                with col3:
                    if 'download_date' in doc_metadata:
                        st.write(f"📅 {doc_metadata['download_date'][:10]}")
            
            if len(metadata) > 10:
                st.write(f"... and {len(metadata) - 10} more documents")
        else:
            st.info("No documents found in the documents folder.")
