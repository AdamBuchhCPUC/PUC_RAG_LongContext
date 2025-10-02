"""
Main entry point for the PUC RAG (LC) System.
Streamlined version for Streamlit Cloud deployment.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix for PyTorch/Streamlit watcher conflict
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()

# Load environment variables from .env file in the script directory
env_path = script_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from: {env_path}")
else:
    # Try loading from current working directory
    load_dotenv()
    print(f"⚠️  .env not found at {env_path}, trying current directory")

# Add src to path for imports
src_path = script_dir / "src"
sys.path.insert(0, str(src_path))

# Debug: Print environment info
print(f"📁 Script directory: {script_dir}")
print(f"📁 Src path: {src_path}")
print(f"🔑 API Key loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

try:
    # Try different import paths for Streamlit Cloud
    try:
        from ui.main_interface import main
    except ImportError:
        from src.ui.main_interface import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required modules are in the src/ directory")
    # Don't exit, let Streamlit handle the error
    import streamlit as st
    st.error(f"❌ Import error: {e}")
    st.error("Make sure all required modules are in the src/ directory")
