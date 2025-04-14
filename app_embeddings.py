"""
app_embeddings.py

This module manages document embedding functionality.
Provides:
- Initialization of the HuggingFace embeddings model
- A pre-loaded embeddings model instance for use throughout the application
- Uses the all-MiniLM-L6-v2 model for creating 384-dimensional embeddings
"""

from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_embeddings_model():
    logger.info("Initializing HuggingFace embeddings model...")
    try:
        # Use the updated import path
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("HuggingFace embeddings model initialized successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embeddings model: {e}")
        st.error(f"Failed to initialize HuggingFace embeddings model: {e}")
        return None

# Pre-load the embeddings model for use throughout the application
embeddings = load_embeddings_model()
