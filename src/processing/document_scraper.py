"""
Document scraping functionality for CPUC proceedings.
Handles web scraping, caching, and document download management.
"""

import streamlit as st

# DEBUG: This file is being loaded
st.info("🔍 [DEBUG] document_scraper.py is being loaded!")
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import requests
from datetime import datetime, timedelta

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Note: Using the same Chrome driver approach as the working rag_test.py


class DocumentCache:
    """Manages caching of downloaded documents and processing state"""
    
    def __init__(self, cache_folder="./cache"):
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(exist_ok=True)
        
        # Add session ID to prevent concurrent user conflicts
        import streamlit as st
        session_id = st.session_state.get('session_id', 'default')
        if 'session_id' not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())[:8]
            session_id = st.session_state.session_id
        
        self.downloads_cache_file = self.cache_folder / f"downloads_cache_{session_id}.json"
        self.processing_cache_file = self.cache_folder / f"processing_cache_{session_id}.json"
        
    def get_downloads_cache(self):
        """Load downloads cache"""
        if self.downloads_cache_file.exists():
            try:
                with open(self.downloads_cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_downloads_cache(self, cache_data):
        """Save downloads cache with file locking for concurrent access"""
        import fcntl
        import tempfile
        
        # Use atomic write to prevent corruption during concurrent access
        temp_file = self.downloads_cache_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            # Atomic move
            temp_file.replace(self.downloads_cache_file)
        except Exception as e:
            # Fallback to original method if atomic write fails
            with open(self.downloads_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
    
    def get_processing_cache(self):
        """Load processing cache"""
        if self.processing_cache_file.exists():
            try:
                with open(self.processing_cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_processing_cache(self, cache_data):
        """Save processing cache with file locking for concurrent access"""
        import tempfile
        
        # Use atomic write to prevent corruption during concurrent access
        temp_file = self.processing_cache_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            # Atomic move
            temp_file.replace(self.processing_cache_file)
        except Exception as e:
            # Fallback to original method if atomic write fails
            with open(self.processing_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
    
    def get_proceeding_cache_key(self, proceeding_number, time_filter, keyword_filter, max_pages):
        """Generate cache key for a proceeding download"""
        cache_key_data = {
            'proceeding': proceeding_number,
            'time_filter': time_filter,
            'keyword_filter': keyword_filter,
            'max_pages': max_pages
        }
        return json.dumps(cache_key_data, sort_keys=True)
    
    def is_proceeding_cached(self, proceeding_number, time_filter, keyword_filter, max_pages):
        """Check if proceeding is already cached with same parameters"""
        cache = self.get_downloads_cache()
        cache_key = self.get_proceeding_cache_key(proceeding_number, time_filter, keyword_filter, max_pages)
        
        if cache_key in cache:
            cache_entry = cache[cache_key]
            # Only use cache if it has documents (don't cache failed attempts)
            if cache_entry.get('documents_count', 0) > 0:
                # For persistent database building, use cache indefinitely
                # Only check if documents were actually found and cached
                return True, cache_entry
        
        return False, None
    
    def cache_proceeding_download(self, proceeding_number, time_filter, keyword_filter, max_pages, documents_count):
        """Cache a proceeding download (only if documents were found)"""
        # Only cache if we actually found documents
        if documents_count > 0:
            cache = self.get_downloads_cache()
            cache_key = self.get_proceeding_cache_key(proceeding_number, time_filter, keyword_filter, max_pages)
            
            cache[cache_key] = {
                'download_time': datetime.now().isoformat(),
                'documents_count': documents_count,
                'proceeding': proceeding_number
            }
            
            self.save_downloads_cache(cache)
            st.info(f"✅ Cached {documents_count} documents for proceeding {proceeding_number}")
        else:
            st.info("ℹ️ No documents found - not caching this attempt")


class CPUCSeleniumScraper:
    """Selenium-based scraper for CPUC documents"""
    
    def __init__(self, headless=True):  # Default to headless for Streamlit Cloud
        self.driver = None
        self.headless = headless
        self._setup_driver()
    
    def _setup_driver(self):
        """Setup Chrome driver using seleniumbase for Streamlit Cloud"""
        try:
            import os
            import sys
            
            # Install Chrome driver using seleniumbase (following the example pattern)
            @st.cache_resource
            def install_chrome():
                os.system('sbase install chromedriver')
                os.system('ln -s /home/appuser/venv/lib/python3.7/site-packages/seleniumbase/drivers/chromedriver /home/appuser/venv/bin/chromedriver')
            
            # Install Chrome driver
            install_chrome()
            
            chrome_options = Options()
            
            # Essential options for Streamlit Cloud
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            chrome_options.add_argument("--log-level=3")
            chrome_options.add_argument("--silent")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            chrome_options.add_argument("--disable-default-apps")
            chrome_options.add_argument("--disable-sync")
            chrome_options.add_argument("--disable-translate")
            chrome_options.add_argument("--hide-scrollbars")
            chrome_options.add_argument("--mute-audio")
            chrome_options.add_argument("--no-first-run")
            chrome_options.add_argument("--disable-background-timer-throttling")
            chrome_options.add_argument("--disable-backgrounding-occluded-windows")
            chrome_options.add_argument("--disable-renderer-backgrounding")
            chrome_options.add_argument("--single-process")
            chrome_options.add_argument("--disable-setuid-sandbox")
            
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_experimental_option("prefs", {
                "profile.default_content_setting_values": {
                    "notifications": 2,
                    "geolocation": 2,
                    "media_stream": 2,
                }
            })
            
            # Use Chrome with seleniumbase-managed driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(30)  # 30 second timeout
            
            # Execute multiple scripts to remove automation indicators
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]})")
            self.driver.execute_script("Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']})")
            self.driver.execute_script("window.chrome = { runtime: {} }")
            
            st.success("Chrome driver initialized successfully with anti-detection measures")
            
        except Exception as e:
            st.error(f"Error initializing Chrome driver: {e}")
            st.error("Troubleshooting steps:")
            st.error("1. Make sure Chrome browser is installed and up to date")
            st.error("2. Check if ChromeDriver is compatible with your Chrome version")
            st.error("3. Try running without headless mode first")
            st.error("4. Check if antivirus software is blocking ChromeDriver")
            st.error("5. For Streamlit Cloud: Make sure packages.txt includes google-chrome-stable and chromedriver")
            raise
    
    def validate_proceeding_number(self, proceeding_number):
        """Validate if a proceeding number exists and is accessible using CPUC docket system"""
        url = f"https://apps.cpuc.ca.gov/apex/f?p=401:56::::RP,57,RIR:P5_PROCEEDING_SELECT:{proceeding_number}"
        
        try:
            st.info(f"Validating proceeding: {proceeding_number}")
            # Add a small delay to make requests look more human-like
            import time
            time.sleep(1)
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 15).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Check for error messages
            page_source = self.driver.page_source.lower()
            current_url = self.driver.current_url.lower()
            
            # Check for specific error patterns (more targeted)
            critical_error_patterns = [
                "sorry, this page isn't available",
                "404",
                "not found",
                "does not exist",
                "no such proceeding",
                "proceeding not found"
            ]
            
            for pattern in critical_error_patterns:
                if pattern in page_source:
                    st.error(f"❌ Proceeding {proceeding_number} not found (detected: '{pattern}')")
                    return False, f"Not found - {pattern}"
            
            # Check for general "error" but be more specific
            if "error" in page_source:
                # Look for specific error contexts
                error_contexts = [
                    "proceeding error",
                    "database error",
                    "system error",
                    "access error",
                    "permission error"
                ]
                
                found_critical_error = False
                for context in error_contexts:
                    if context in page_source:
                        st.error(f"❌ Proceeding {proceeding_number} has {context}")
                        return False, f"Critical error - {context}"
            
            # Check if we're redirected to an error page
            if "error" in current_url or "404" in current_url:
                st.error(f"❌ Proceeding {proceeding_number} redirected to error page")
                return False, "Redirected to error page"
            
            # Check if we can find proceeding details
            try:
                WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
                
                # Additional check: look for document links or tables
                try:
                    # Look for any links that might be documents
                    links = self.driver.find_elements(By.TAG_NAME, "a")
                    document_links = [link for link in links if link.get_attribute("href") and ".pdf" in link.get_attribute("href").lower()]
                    
                    if document_links:
                        st.success(f"✅ Proceeding {proceeding_number} validated successfully (found {len(document_links)} document links)")
                        return True, "Valid with documents"
                    else:
                        st.warning(f"⚠️ Proceeding {proceeding_number} exists but no documents found yet")
                        return True, "Valid but no documents"
                        
                except Exception as e:
                    st.warning(f"⚠️ Could not check for documents: {e}")
                    st.success(f"✅ Proceeding {proceeding_number} validated successfully")
                    return True, "Valid"
                    
            except TimeoutException:
                st.warning(f"⚠️ Could not confirm proceeding {proceeding_number} details")
                return False, "Details not found"
                
        except Exception as e:
            st.error(f"❌ Error validating proceeding {proceeding_number}: {e}")
            return False, str(e)
    
    def navigate_to_documents(self, proceeding_number):
        """Navigate to the documents tab with improved error handling"""
        st.info(f"🔍 [navigate_to_documents] Starting navigation for proceeding {proceeding_number}")
        try:
            # First try to click on the Documents tab instead of direct navigation
            st.info("🔍 Attempting to click on Documents tab...")
            
            # Look for the Documents tab link
            documents_tab = self.driver.find_element(By.XPATH, "//a[contains(@href, 'f?p=401:57')]")
            if documents_tab:
                st.info("✅ Found Documents tab, clicking...")
                documents_tab.click()
                
                # Wait for page to load
                WebDriverWait(self.driver, 15).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
                
                st.info("✅ Documents page loaded")
                return True
                
        except Exception as e:
            st.warning(f"⚠️ Could not click Documents tab: {e}")
            st.info("🔄 Falling back to direct URL navigation...")
            
            # Fallback to direct navigation
            url = f"https://apps.cpuc.ca.gov/apex/f?p=401:57::::RP,57,RIR:P5_PROCEEDING_SELECT:{proceeding_number}"
            
            try:
                # Add a small delay to make requests look more human-like
                import time
                time.sleep(1)
                self.driver.get(url)
                
                # Wait for page to load and check for common error messages
                WebDriverWait(self.driver, 15).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
                
            except Exception as e:
                st.error(f"❌ Error navigating to documents page: {e}")
                return False
            
            # Check if we got an error page
            page_source = self.driver.page_source.lower()
            if "sorry, this page isn't available" in page_source:
                st.error("❌ CPUC website returned 'Sorry, this page isn't available'")
                return False
            
            if "404" in page_source or "not found" in page_source:
                st.error(f"❌ Proceeding {proceeding_number} not found (404 error)")
                return False
            
            # Try to find the documents table
            try:
                # Wait for the documents table to appear
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table.a-IRR-table"))
                )
                st.success("✅ Documents page loaded successfully")
                return True
            except TimeoutException:
                st.warning("⚠️ Documents table not found. This might be a proceeding with no documents.")
                return False
                    
        except TimeoutException as e:
            st.error(f"❌ Timeout navigating to documents page: {e}")
            return False
        except Exception as e:
            st.error(f"❌ Error navigating to documents: {e}")
            return False
    
    def get_documents_from_current_page(self, filter_intervenor_comp=True, keyword_filter=None):
        """Extract documents from the currently displayed page using the same approach as rag_test.py"""
        st.info(f"🔍 [get_documents_from_current_page] Starting with filter_intervenor_comp={filter_intervenor_comp}, keyword_filter={keyword_filter}")
        documents = []
        
        try:
            table = self.driver.find_element(By.CSS_SELECTOR, "table.a-IRR-table")
            rows = table.find_elements(By.TAG_NAME, "tr")[1:]
            
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 4:
                    filing_date = cells[0].text.strip()
                    
                    doc_type_cell = cells[1]
                    try:
                        doc_link = doc_type_cell.find_element(By.TAG_NAME, "a")
                        doc_type = doc_link.text.strip()
                        doc_url = doc_link.get_attribute("href")
                    except:
                        doc_type = doc_type_cell.text.strip()
                        doc_url = None
                    
                    filed_by = cells[2].text.strip()
                    description = cells[3].text.strip()
                    
                    # Filter out ex parte documents, certificates of service, and intervenor compensation at the link collection stage
                    # This prevents unnecessary clicks to document detail pages
                    doc_type_lower = doc_type.lower()
                    description_lower = description.lower()
                    
                    # Skip ex parte documents
                    if (doc_type_lower == 'exparte' or 
                        'notice of ex parte' in description_lower or 
                        'notice of exparte' in description_lower or
                        'exparte' in description_lower):
                        continue
                    
                    # Skip certificates of service
                    if ('certificate of service' in description_lower or 
                        'certificate of service' in doc_type_lower or
                        '(cos)' in description_lower):
                        continue
                    
                    # Skip intervenor compensation documents (configurable)
                    if filter_intervenor_comp and 'intervenor compensation' in description_lower:
                        continue
                    
                    documents.append({
                        'filing_date': filing_date,
                        'document_type': doc_type,
                        'filed_by': filed_by,
                        'description': description,
                        'document_url': doc_url
                    })
            
            # Apply keyword filter if specified
            if keyword_filter and keyword_filter in ["PROPOSED DECISION", "SCOPING RULING", "SCOPING MEMO", "DECISION", "RULING"]:
                st.info(f"🔍 [get_documents_from_current_page] Applying keyword filter: {keyword_filter}")
                st.info(f"📄 [get_documents_from_current_page] Starting with {len(documents)} documents")
                st.info(f"🔍 [get_documents_from_current_page] Calling _apply_keyword_filter")
                filtered_documents = self._apply_keyword_filter(documents, keyword_filter)
                st.info(f"📄 [get_documents_from_current_page] _apply_keyword_filter returned {len(filtered_documents)} documents (was {len(documents)})")
                
                # Debug: Show what documents are being kept
                if len(filtered_documents) > 0:
                    st.info(f"🔍 All {len(filtered_documents)} documents after filtering:")
                    
                    # Create a detailed table
                    import pandas as pd
                    
                    # Prepare data for the table
                    table_data = []
                    for i, doc in enumerate(filtered_documents):
                        table_data.append({
                            'Index': i + 1,
                            'Document Type': doc.get('document_type', 'Unknown'),
                            'Filing Date': doc.get('filing_date', 'Unknown date'),
                            'Filed By': doc.get('filed_by', 'Unknown'),
                            'Description': doc.get('description', 'No description')[:50] + '...' if len(doc.get('description', '')) > 50 else doc.get('description', 'No description')
                        })
                    
                    # Create and display the table
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Also show a summary
                    st.info(f"📊 Summary: {len(filtered_documents)} documents will be downloaded")
                
                return filtered_documents
            
            return documents
        except Exception as e:
            st.error(f"Error extracting documents: {e}")
            return []
    
    def _apply_keyword_filter(self, documents, keyword_filter):
        """Apply keyword filter to documents - include target document type and everything after it"""
        st.info(f"🔍 [_apply_keyword_filter] Starting with {len(documents)} documents, looking for '{keyword_filter}'")
        from datetime import datetime
        
        # Find the last occurrence of the target document type
        st.info(f"🔍 [_apply_keyword_filter] Calling find_last_document_type_date for '{keyword_filter}'")
        last_document_date = self.find_last_document_type_date(documents, keyword_filter)
        st.info(f"🔍 [_apply_keyword_filter] find_last_document_type_date returned: {last_document_date}")
        
        if not last_document_date:
            st.warning(f"⚠️ No {keyword_filter} found in documents. Using all documents.")
            return documents
        
        st.info(f"📅 Found last {keyword_filter} on: {last_document_date.strftime('%B %d, %Y')}")
        st.info(f"🛑 Will include documents UP TO AND INCLUDING the most recent {keyword_filter}")
        
        # Debug: Show all documents and their dates
        st.info(f"🔍 All {len(documents)} documents before filtering:")
        
        # Create a table of all documents
        import pandas as pd
        
        # Prepare data for the table
        table_data = []
        for i, doc in enumerate(documents):
            table_data.append({
                'Index': i + 1,
                'Document Type': doc.get('document_type', 'Unknown'),
                'Filing Date': doc.get('filing_date', 'Unknown date'),
                'Filed By': doc.get('filed_by', 'Unknown'),
                'Description': doc.get('description', 'No description')[:50] + '...' if len(doc.get('description', '')) > 50 else doc.get('description', 'No description')
            })
        
        # Create and display the table
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        filtered_documents = []
        filtered_out_count = 0
        
        for doc in documents:
            # Parse the document date
            doc_date = None
            filing_date_str = doc['filing_date'].strip()
            
            # Try multiple date formats
            date_formats = [
                "%B %d, %Y",      # August 28, 2025
                "%B %d,%Y",       # August 28,2025 (no space after comma)
                "%B %d, %Y",      # August 28, 2025 (with extra spaces)
                "%m/%d/%Y",       # 08/28/2025
                "%m-%d-%Y",       # 08-28-2025
            ]
            
            for date_format in date_formats:
                try:
                    doc_date = datetime.strptime(filing_date_str, date_format)
                    break
                except ValueError:
                    continue
            
            if doc_date is None:
                # If we can't parse the date, include the document
                filtered_documents.append(doc)
                continue
            
            # Include documents up to and including the last occurrence of the target document type
            if doc_date <= last_document_date:
                filtered_documents.append(doc)
                st.write(f"  ✅ KEEPING: {doc.get('document_type', 'Unknown')} - {doc.get('filing_date', 'Unknown date')} (date: {doc_date.strftime('%B %d, %Y')})")
            else:
                filtered_out_count += 1
                st.write(f"  ❌ FILTERING OUT: {doc.get('document_type', 'Unknown')} - {doc.get('filing_date', 'Unknown date')} (date: {doc_date.strftime('%B %d, %Y')})")
        
        st.info(f"🛑 Filtered out {filtered_out_count} documents after {last_document_date.strftime('%B %d, %Y')}")
        return filtered_documents
    
    def find_last_document_type_date(self, documents, document_type):
        """Find the date of the most recent document of a specific type"""
        st.info(f"🔍 [find_last_document_type_date] Starting with {len(documents)} documents, looking for '{document_type}'")
        from datetime import datetime
        
        # Debug: Show all document types found
        st.info(f"🔍 Looking for document type: '{document_type}'")
        st.info(f"🔍 All document types found in the documents:")
        document_types = set()
        for doc in documents:
            doc_type = doc.get('document_type', 'Unknown')
            document_types.add(doc_type)
        
        for doc_type in sorted(document_types):
            st.write(f"  - '{doc_type}'")
        
        document_dates = []
        
        for doc in documents:
            doc_type = doc.get('document_type', '')
            st.info(f"🔍 [find_last_document_type_date] Checking document type: '{doc_type}' == '{document_type}'? {doc_type.upper() == document_type.upper()}")
            if doc_type.upper() == document_type.upper():
                filing_date_str = doc['filing_date'].strip()
                
                # Try multiple date formats
                date_formats = [
                    "%B %d, %Y",      # August 28, 2025
                    "%B %d,%Y",       # August 28,2025 (no space after comma)
                    "%B %d, %Y",      # August 28, 2025 (with extra spaces)
                    "%m/%d/%Y",       # 08/28/2025
                    "%m-%d-%Y",       # 08-28-2025
                ]
                
                doc_date = None
                for date_format in date_formats:
                    try:
                        doc_date = datetime.strptime(filing_date_str, date_format)
                        break
                    except ValueError:
                        continue
            
                if doc_date is not None:
                    document_dates.append(doc_date)
        
        if document_dates:
            return max(document_dates)
        return None

    def get_all_documents(self, proceeding_number, max_pages=None, time_filter=None, keyword_filter=None, filter_intervenor_comp=True):
        """Get all documents for a proceeding using the same approach as rag_test.py"""
        st.info(f"🔍 [get_all_documents] Starting with proceeding={proceeding_number}, max_pages={max_pages}, time_filter={time_filter}, keyword_filter={keyword_filter}")
        from datetime import datetime, timedelta
        documents = []
        
        try:
            # Navigate to documents page
            if not self.navigate_to_documents(proceeding_number):
                return documents
            
            # Get documents from current page
            st.info(f"🔍 [get_all_documents] Calling get_documents_from_current_page with filter_intervenor_comp={filter_intervenor_comp}, keyword_filter={keyword_filter}")
            page_documents = self.get_documents_from_current_page(filter_intervenor_comp, keyword_filter)
            st.info(f"🔍 [get_all_documents] get_documents_from_current_page returned {len(page_documents)} documents")
            documents.extend(page_documents)
            
            # Apply time filter if specified (keyword filter is now handled in get_documents_from_current_page)
            # BUT ONLY if no keyword filter was applied (to avoid overriding keyword filtering)
            if time_filter and time_filter != "Whole docket" and not keyword_filter:
                st.info(f"🔍 Applying time filter: {time_filter}")
                filtered_documents = []
                filtered_out_count = 0
                
                for doc in documents:
                    include_doc = True
                    
                    if doc.get('filing_date'):
                        try:
                            # Parse the document date
                            doc_date = None
                            try:
                                doc_date = datetime.strptime(doc['filing_date'], "%B %d, %Y")
                            except:
                                try:
                                    doc_date = datetime.strptime(doc['filing_date'], "%m/%d/%Y")
                                except:
                                    continue
                            
                            if time_filter == "Last 30 days" and (datetime.now() - doc_date).days > 30:
                                include_doc = False
                            elif time_filter == "Last 60 days" and (datetime.now() - doc_date).days > 60:
                                include_doc = False
                            elif time_filter == "Last 90 days" and (datetime.now() - doc_date).days > 90:
                                include_doc = False
                            elif time_filter == "Last 180 days" and (datetime.now() - doc_date).days > 180:
                                include_doc = False
                            elif time_filter == "Last 12 months" and (datetime.now() - doc_date).days > 365:
                                include_doc = False
                            elif time_filter == "Since 2020" and doc_date.year < 2020:
                                include_doc = False
                            elif time_filter == "Since 2019" and doc_date.year < 2019:
                                include_doc = False
                            elif time_filter == "Since 2018" and doc_date.year < 2018:
                                include_doc = False
                            elif time_filter == "Since 2017" and doc_date.year < 2017:
                                include_doc = False
                            elif time_filter == "Since 2016" and doc_date.year < 2016:
                                include_doc = False
                            elif time_filter == "Since 2015" and doc_date.year < 2015:
                                include_doc = False
                            elif time_filter == "Since 2014" and doc_date.year < 2014:
                                include_doc = False
                            elif time_filter == "Since 2013" and doc_date.year < 2013:
                                include_doc = False
                            elif time_filter == "Since 2012" and doc_date.year < 2012:
                                include_doc = False
                            elif time_filter == "Since 2011" and doc_date.year < 2011:
                                include_doc = False
                            elif time_filter == "Since 2010" and doc_date.year < 2010:
                                include_doc = False
                        except:
                            pass
                    
                    if include_doc:
                        filtered_documents.append(doc)
                    else:
                        filtered_out_count += 1
                
                documents = filtered_documents
                st.info(f"🛑 Time filter removed {filtered_out_count} documents")
                st.info(f"📄 {len(documents)} documents remain after time filtering")
            
            # Apply max_pages limit - this should limit to first N pages of documents
            if max_pages and max_pages > 0:
                # Estimate documents per page (typically 10-20 documents per page)
                estimated_docs_per_page = 15
                max_documents = max_pages * estimated_docs_per_page
                st.info(f"🔍 Document limit check: {len(documents)} documents found, max allowed: {max_documents} (max_pages={max_pages})")
                
                if len(documents) > max_documents:
                    st.warning(f"⚠️ TOO MANY DOCUMENTS! Limiting to {max_documents} documents (max_pages={max_pages})")
                    st.warning(f"⚠️ Original count: {len(documents)} documents")
                    documents = documents[:max_documents]
                    st.success(f"✅ Limited to {len(documents)} documents")
                else:
                    st.info(f"✅ Document count ({len(documents)}) is within limit ({max_documents})")
                
        except Exception as e:
            st.error(f"Error scraping documents: {e}")
        
        return documents
    
    def scrape_proceeding(self, proceeding_number, time_filter="Whole docket", keyword_filter="None", max_pages=10):
        """Scrape a CPUC proceeding for documents using the full CPUC docket system"""
        st.info(f"🔍 [scrape_proceeding] Starting with proceeding={proceeding_number}, time_filter={time_filter}, keyword_filter={keyword_filter}, max_pages={max_pages}")
        try:
            # Validate proceeding first
            is_valid, message = self.validate_proceeding_number(proceeding_number)
            if not is_valid:
                st.error(f"❌ {message}")
                return 0
            
            st.info(f"✅ {message}")
            
            # Navigate to documents page
            if not self.navigate_to_documents(proceeding_number):
                st.error("❌ Could not navigate to documents page")
                return 0
            
            # Get all documents with proper filtering and pagination
            st.info(f"🔍 [scrape_proceeding] Calling get_all_documents with max_pages={max_pages}, time_filter={time_filter}, keyword_filter={keyword_filter}")
            all_documents = self.get_all_documents(proceeding_number, max_pages, time_filter, keyword_filter)
            
            if not all_documents:
                st.warning(f"No documents found for proceeding {proceeding_number}")
                return 0
            
            st.info(f"📄 Found {len(all_documents)} documents")
            
            # Download documents
            st.info(f"🔍 [scrape_proceeding] Calling _download_documents with {len(all_documents)} documents")
            downloaded_count = self._download_documents(all_documents, proceeding_number)
            st.info(f"🔍 [scrape_proceeding] _download_documents returned {downloaded_count} downloaded documents")
            return downloaded_count
                
        except Exception as e:
            st.error(f"Error scraping proceeding {proceeding_number}: {e}")
            return 0
    
    def _extract_documents(self, proceeding_number, time_filter, keyword_filter, max_pages):
        """Extract document information from the proceeding page"""
        documents = []
        
        try:
            # Find document table or list
            # This is a simplified version - the actual implementation would be more complex
            doc_elements = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='.pdf']")
            
            for element in doc_elements[:max_pages * 10]:  # Limit to reasonable number
                try:
                    href = element.get_attribute('href')
                    text = element.text.strip()
                    
                    if href and text:
                        documents.append({
                            'url': href,
                            'title': text,
                            'proceeding': proceeding_number
                        })
                except:
                    continue
            
            return documents
            
        except Exception as e:
            st.error(f"Error extracting documents: {e}")
            return []
    
    
    def _download_documents(self, documents, proceeding_number):
        """Download documents to the documents folder with full metadata"""
        st.info(f"🔍 [_download_documents] Starting with {len(documents)} documents for proceeding {proceeding_number}")
        documents_folder = Path("./documents")
        documents_folder.mkdir(exist_ok=True)
        
        # Create metadata folder
        metadata_folder = documents_folder / "metadata"
        metadata_folder.mkdir(exist_ok=True)
        
        # Create proceeding subfolder
        proceeding_folder = documents_folder / proceeding_number
        proceeding_folder.mkdir(exist_ok=True)
        
        downloaded_count = 0
        
        for i, doc in enumerate(documents):
            try:
                # Get PDF URL from document detail page if available
                pdf_url = None
                if doc.get('document_url'):
                    try:
                        # Add Chrome crash recovery
                        try:
                            pdf_links = self.scrape_document_detail_page(doc['document_url'])
                            if pdf_links and len(pdf_links) > 0 and 'error' not in pdf_links[0]:
                                pdf_url = pdf_links[0].get('PDF URL', '')
                        except Exception as e:
                            st.warning(f"⚠️ Chrome crashed while scraping {doc.get('document_type', 'Unknown')}: {e}")
                            # Try to restart Chrome driver
                            try:
                                self.cleanup()
                                self._setup_driver()
                                st.info("🔄 Chrome driver restarted")
                            except:
                                st.error("❌ Could not restart Chrome driver")
                                continue
                    except:
                        pass
                
                if not pdf_url:
                    st.warning(f"⚠️ No PDF URL found for {doc.get('document_type', 'Unknown')}")
                    continue
                
                # Download PDF
                response = requests.get(pdf_url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Create filename with same convention as original
                import re
                from processing.document_downloader import create_acronym, get_document_suffix
                
                safe_filename = re.sub(r'[<>:"/\\|?*]', '_', doc['document_type'])[:100]
                filing_date = doc.get('filing_date', '').replace('/', '-')
                submitter_acronym = create_acronym(doc.get('filed_by', 'Unknown'))
                
                # Get document suffix for multiple documents
                doc_suffix = get_document_suffix(
                    doc.get('document_type', ''), 
                    doc.get('description', ''), 
                    i
                )
                
                # Create filename with same convention as original
                if doc_suffix == "docA" and len(documents) == 1:
                    # Single document, no suffix needed
                    filename = f"{proceeding_number}_{submitter_acronym}_{filing_date}_{safe_filename}_{i}.pdf"
                else:
                    # Multiple documents, add suffix
                    filename = f"{proceeding_number}_{submitter_acronym}_{filing_date}_{safe_filename}_{doc_suffix}_{i}.pdf"
                
                filepath = proceeding_folder / filename
                
                # Save file
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Create comprehensive metadata (same as original)
                metadata = {
                    'filename': filename,
                    'original_title': doc.get('title', ''),
                    'filing_date': doc.get('filing_date', ''),
                    'document_type': doc.get('document_type', ''),
                    'filed_by': doc.get('filed_by', ''),
                    'description': doc.get('description', ''),
                    'pdf_url': pdf_url,
                    'document_url': doc.get('document_url', ''),
                    'proceeding': proceeding_number,
                    'page_number': doc.get('page_number', 1),
                    'download_date': datetime.now().isoformat(),
                    'doc_suffix': doc_suffix,
                    'docket_group': f"{proceeding_number}_{i}",
                    'inverse_index_created': False,  # Track inverse index processing status
                    'processed': False
                }
                
                # Save metadata
                metadata_file = metadata_folder / f"{filename}.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                downloaded_count += 1
                st.success(f"✅ Downloaded: {filename}")
                
            except Exception as e:
                st.error(f"❌ Failed to download {doc.get('document_type', 'Unknown')}: {e}")
                continue
        
        return downloaded_count
    
    def scrape_document_detail_page(self, detail_url, retry_count=0):
        """Scrape a document detail page to get PDF links and title"""
        max_retries = 2
        
        try:
            # Set a shorter timeout for individual page loads
            self.driver.set_page_load_timeout(10)  # Reduced timeout
            self.driver.get(detail_url)
            
            # Wait for page to load with shorter timeout
            WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        except TimeoutException:
            if retry_count < max_retries:
                st.warning(f"⚠️ Timeout loading detail page (attempt {retry_count + 1}/{max_retries + 1})")
                import time
                time.sleep(1)  # Shorter wait
                return self.scrape_document_detail_page(detail_url, retry_count + 1)
            else:
                st.warning(f"⚠️ Timeout loading detail page after {max_retries + 1} attempts")
                return [{"error": "timeout", "url": detail_url}]
        except Exception as e:
            st.warning(f"Error loading detail page: {e}")
            return [{"error": str(e), "url": detail_url}]
        
        title = ""
        
        # Extract title with error handling
        try:
            title_cells = self.driver.find_elements(By.CSS_SELECTOR, "td.ResultTitleTD")
            if title_cells:
                title_text = title_cells[0].text.strip()
                if 'Proceeding:' in title_text:
                    title = title_text.split('Proceeding:')[0].strip()
                else:
                    title = title_text
        except Exception as e:
            st.warning(f"Error extracting title: {e}")
        
        # Look for PDF links with enhanced error handling
        pdf_links = []
        seen_urls = set()
        
        try:
            # Use shorter wait time for finding elements
            result_table = WebDriverWait(self.driver, 3).until(
                EC.presence_of_element_located((By.ID, "ResultTable"))
            )
            rows = result_table.find_elements(By.TAG_NAME, "tr")
            
            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 4:
                        title_cell = cells[0]
                        if 'ResultTitleTD' in title_cell.get_attribute("class") or "":
                            row_title_text = title_cell.text.strip()
                            if 'Proceeding:' in row_title_text:
                                row_title = row_title_text.split('Proceeding:')[0].strip()
                            else:
                                row_title = title
                        else:
                            row_title = title
                        
                        link_cell = cells[2]
                        if 'ResultLinkTD' in link_cell.get_attribute("class") or "":
                            links = link_cell.find_elements(By.TAG_NAME, "a")
                            for link in links:
                                try:
                                    link_text = link.text.strip()
                                    link_href = link.get_attribute("href")
                                    
                                    if (link_text.upper().find('PDF') != -1 or 
                                        (link_href and link_href.lower().find('.pdf') != -1)):
                                        
                                        if link_href and link_href not in seen_urls:
                                            seen_urls.add(link_href)
                                            pdf_links.append({
                                                'Title': row_title,
                                                'PDF Link Text': link_text,
                                                'PDF URL': link_href
                                            })
                                except Exception as e:
                                    # Skip problematic links
                                    continue
                except Exception as e:
                    # Skip problematic rows
                    continue
        except Exception as e:
            st.warning(f"Error extracting PDF links: {e}")
            # Return empty list instead of crashing
            return []
        
        return pdf_links
    
    def cleanup(self):
        """Clean up the driver"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
