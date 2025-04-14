"""
app_rag.py

This module orchestrates the RAG (Retrieval Augmented Generation) functionality.
It serves as the central coordinator between document processing, retrieval, and LLM components,
managing the entire pipeline from user query to final response.

Key functions:
- process_user_query: Main entry point that processes a user query using the RAG pipeline.
  It retrieves relevant documents, fetches parent context, formats the context for the LLM,
  and returns a comprehensive answer with metrics.

The RAG pipeline flow:
1. User submits query → process_user_query
2. Vector store retrieval via app_vector.py
3. Hybrid document retrieval via app_retrieval.py (combining vector and SQL search)
4. Parent context fetching for better understanding
5. Context formatting for the LLM
6. LLM response generation via app_llm.py
7. Return of answer with context and metrics
"""

import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from app_retrieval import (
    implement_parent_document_retriever,
    fetch_parent_context,
    hybrid_retriever,
    DocumentCollection
)
from app_vector import get_vector_store
from app_llm import get_llm_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_user_query(query: str, doc_name: str, limit: int = 30) -> Dict[str, Any]:
    """
    Process a user query using the RAG pipeline
    
    Args:
        query: User's question
        doc_name: Document to search in
        limit: Maximum number of results to return
        
    Returns:
        dict: Results including answer and context
    """
    try:
        logger.info(f"Processing query: '{query}' for document: {doc_name}")
        
        # Get vector store
        vector_store = get_vector_store()
        if not vector_store:
            logger.error("Failed to get vector store")
            return {
                "error": "Failed to initialize vector store",
                "answer": None,
                "context": None,
                "metrics": None
            }
            
        # Use hybrid retriever to get relevant documents
        results = hybrid_retriever(
            query=query,
            vector_store=vector_store,
            doc_name=doc_name,
            limit=limit
        )
        
        if not results or len(results) == 0:
            logger.warning("No results found for query")
            return {
                "error": "No relevant documents found",
                "answer": None,
                "context": None,
                "metrics": None
            }
            
        # Get parent context for better understanding
        context_docs = fetch_parent_context(results)
        
        # Format context for LLM
        context = format_retrieval_context(context_docs)
        
        # Get answer from LLM
        answer = get_llm_response(query=query, context=context)
        
        # Get metrics
        metrics = context_docs.get_metrics()
        
        return {
            "error": None,
            "answer": answer,
            "context": context,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "error": str(e),
            "answer": None,
            "context": None,
            "metrics": None
        }

from app_functions import (
    format_retrieval_context,
    create_prompt_with_context,
    extract_and_cite_answer,
    format_metrics_for_streamlit
)
