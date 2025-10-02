# PUC RAG (LC) - Streamlined CPUC Document Q&A System

A streamlined version of the CPUC Document Q&A System optimized for Streamlit Cloud deployment.

## Features

- **Intelligent Document Download**: Download CPUC proceedings with time-based filtering and smart stopping conditions
- **Hierarchical Processing**: Advanced text chunking with content analysis
- **Hybrid Search**: Vector similarity + BM25 keyword search
- **Model Flexibility**: Support for GPT-4o, GPT-5, O1, O3, and other models
- **Smart Caching**: Content-hash caching to prevent re-embedding
- **Cost Tracking**: Real-time cost monitoring and optimization
- **Web Scraping**: Automated document discovery and download from CPUC website

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

## Deployment to Streamlit Cloud

1. **Push to GitHub**: Upload this repository to GitHub
2. **Deploy**: Use Streamlit Cloud to deploy from GitHub
3. **Set Secrets**: Add `OPENAI_API_KEY` in Streamlit Cloud secrets

## Architecture

```
PUC RAG (LC)/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── README.md                 # This file
└── src/
    ├── processing/           # Document processing
    ├── analysis/            # QA system and models
    └── ui/                  # Streamlit interface
```

## Key Components

- **Document Processor**: Handles PDF extraction and hierarchical chunking
- **QA System**: Hybrid search with model-aware chunk selection
- **Model Config**: Complete model support with TPM limits and pricing
- **Cache Manager**: Content-hash caching for cost optimization

## Model Support

Supports all OpenAI models including:
- GPT-4o, GPT-4o-mini
- GPT-5, GPT-5-mini, GPT-5-nano
- O1, O1-mini, O3, O3-mini
- O4-mini and other reasoning models

Each model has optimized TPM limits, pricing, and chunking strategies.
