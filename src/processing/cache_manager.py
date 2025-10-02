"""
Content-hash caching functionality.
Handles caching to skip re-embedding unchanged chunks.
"""

import streamlit as st
import sqlite3
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain.schema import Document


class ContentHashCache:
    """Content-hash caching to skip re-embedding unchanged chunks"""
    
    def __init__(self, cache_db_path: str = "./processed_data_vector/embedding_cache.db"):
        self.cache_db_path = cache_db_path
        self._init_cache_db()
    
    def _init_cache_db(self):
        """Initialize the cache database"""
        try:
            # Ensure directory exists
            Path(self.cache_db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Create cache table
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_hash TEXT PRIMARY KEY,
                    embedding BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            st.warning(f"Could not initialize cache database: {e}")
    
    def get_cached_embedding(self, content: str) -> Tuple[bool, Any, Dict]:
        """Get cached embedding for content if it exists"""
        try:
            content_hash = self._hash_content(content)
            
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute(
                'SELECT embedding, metadata FROM embedding_cache WHERE content_hash = ?',
                (content_hash,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                embedding, metadata_json = result
                metadata = json.loads(metadata_json) if metadata_json else {}
                return True, embedding, metadata
            else:
                return False, None, {}
                
        except Exception as e:
            st.warning(f"Error checking cache: {e}")
            return False, None, {}
    
    def cache_embedding(self, content: str, embedding: Any, metadata: Dict):
        """Cache embedding for content"""
        try:
            content_hash = self._hash_content(content)
            
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO embedding_cache 
                (content_hash, embedding, metadata) 
                VALUES (?, ?, ?)
            ''', (content_hash, embedding, json.dumps(metadata)))
            conn.commit()
            conn.close()
            
        except Exception as e:
            st.warning(f"Error caching embedding: {e}")
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM embedding_cache')
            conn.commit()
            conn.close()
            st.success("✅ Cache cleared successfully")
        except Exception as e:
            st.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM embedding_cache')
            count = cursor.fetchone()[0]
            conn.close()
            
            return {
                'cached_embeddings': count,
                'cache_size_mb': Path(self.cache_db_path).stat().st_size / (1024 * 1024) if Path(self.cache_db_path).exists() else 0
            }
        except Exception as e:
            st.warning(f"Error getting cache stats: {e}")
            return {'cached_embeddings': 0, 'cache_size_mb': 0}
