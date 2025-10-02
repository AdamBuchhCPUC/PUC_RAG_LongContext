# PUC RAG (LC) - Deployment Guide

## Streamlit Cloud Deployment

### 1. Prepare Repository

1. **Push to GitHub**: Upload this repository to GitHub
2. **Set Repository to Public**: Streamlit Cloud requires public repositories for free deployment

### 2. Deploy to Streamlit Cloud

1. **Visit Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io)
2. **Sign in with GitHub**: Use your GitHub account
3. **Deploy New App**: Click "New app"
4. **Repository**: Select your GitHub repository
5. **Branch**: Select `main` branch
6. **Main file path**: Enter `main.py`

### 3. Configure Secrets

In Streamlit Cloud, add the following secrets:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

### 4. Environment Variables

The app will automatically load from `.env` file if present, or use Streamlit Cloud secrets.

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Application

```bash
streamlit run main.py
```

## Features

### ✅ Implemented
- **Document Download**: Manual PDF download and organization
- **Hierarchical Processing**: Advanced text chunking with content analysis
- **Hybrid Search**: Vector similarity + BM25 keyword search
- **Model Support**: GPT-4o, GPT-5, O1, O3, and other models
- **Smart Caching**: Content-hash caching to prevent re-embedding
- **Cost Tracking**: Real-time cost monitoring
- **Streamlined UI**: 3-tab interface (Download, Process, Ask Questions)

### 🚀 Key Improvements
- **Simplified Architecture**: Removed complex parallel processing
- **Essential Components Only**: Kept only necessary features
- **Deployment Ready**: Optimized for Streamlit Cloud
- **Clean Code**: Removed unused imports and dependencies

## File Structure

```
PUC RAG (LC)/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── README.md                 # Documentation
├── DEPLOYMENT.md             # This file
└── src/
    ├── processing/           # Document processing
    │   ├── document_downloader.py
    │   ├── document_processor.py
    │   ├── hierarchical_chunker.py
    │   └── cache_manager.py
    ├── analysis/            # QA system and models
    │   ├── qa_system.py
    │   ├── model_config.py
    │   └── llm_utils.py
    └── ui/                  # Streamlit interface
        ├── main_interface.py
        ├── sidebar.py
        ├── results_display.py
        └── tabs/
            ├── download_tab.py
            ├── processing_tab.py
            └── qa_tab.py
```

## Usage

1. **Download Documents**: Use the Download tab to add PDFs
2. **Process Documents**: Use the Process tab to convert PDFs to searchable text
3. **Ask Questions**: Use the Ask Questions tab to query your documents

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **API Key Issues**: Check that OPENAI_API_KEY is set correctly
3. **Memory Issues**: For large documents, consider using smaller chunk sizes
4. **Rate Limits**: Use models with higher TPM limits for large datasets
5. **Web Scraping Issues**: Make sure Chrome is installed for document downloading
6. **ChromeDriver Issues**: ChromeDriver is downloaded automatically via webdriver-manager

### Support

For issues or questions, check the original RAG Test repository for advanced features and troubleshooting guides.
