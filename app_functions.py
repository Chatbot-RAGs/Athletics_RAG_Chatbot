"""
app_functions.py

This module provides utility functions used across the application.
"""

import os
import logging
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_retrieval_context(docs: List[Document]) -> str:
    """
    Format retrieved documents into a context string for the LLM
    
    Args:
        docs: Collection of retrieved documents
        
    Returns:
        str: Formatted context string
    """
    try:
        context_parts = []
        
        # First add non-parent documents
        for doc in docs:
            if not doc.metadata.get("is_parent", False):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                context_parts.append(f"\nSource: {source}, Page: {page}\nContent: {doc.page_content}")
        
        # Then add parent documents for additional context
        for doc in docs:
            if doc.metadata.get("is_parent", False):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                child_page = doc.metadata.get("parent_of_page", "Unknown")
                context_parts.append(f"\nParent Context (Source: {source}, Page: {page}, Context for Page: {child_page}):\n{doc.page_content}")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Error formatting context: {str(e)}")
        # Return whatever we can from the docs
        return "\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content'))

def create_prompt_with_context(query: str, context: str) -> str:
    """
    Create a prompt that includes the context and query
    
    Args:
        query: User's question
        context: Retrieved context
        
    Returns:
        str: Complete prompt for LLM
    """
    return f"""
Please answer the following question based on the provided context. 
If the answer cannot be fully determined from the context, say so.

Context:
{context}

Question: {query}

Answer:"""

def extract_and_cite_answer(response: str, context_docs: List[Document]) -> Dict[str, Any]:
    """
    Extract answer and add citations from source documents
    
    Args:
        response: LLM response
        context_docs: Source documents
        
    Returns:
        dict: Answer with citations
    """
    try:
        # Extract sources
        sources = []
        for doc in context_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            sources.append({"source": source, "page": page})
        
        return {
            "answer": response,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error extracting citations: {str(e)}")
        return {
            "answer": response,
            "sources": []
        }

def format_metrics_for_streamlit(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metrics for display in Streamlit
    
    Args:
        metrics: Raw metrics from retrieval
        
    Returns:
        dict: Formatted metrics for display
    """
    try:
        return {
            "Total Results": metrics.get("total_results", 0),
            "Vector Results": metrics.get("vector_results", 0),
            "SQL Results": metrics.get("sql_results", 0),
            "Table Results": metrics.get("table_results", 0),
            "Parent Documents": metrics.get("parent_count", 0)
        }
    except Exception as e:
        logger.error(f"Error formatting metrics: {str(e)}")
        return {}

def save_dataframe(df: pd.DataFrame, query: str, query_type: str = "nl_query") -> Optional[str]:
    """
    Save a dataframe to a parquet file in the temp_data directory
    
    Args:
        df: DataFrame to save
        query: Query that generated the data
        query_type: Type of query (nl_query or mysql_nl_query)
        
    Returns:
        str: Path to saved file, or None if save failed
    """
    try:
        # Create temp_data directory if it doesn't exist
        temp_dir = Path("temp_data")
        temp_dir.mkdir(exist_ok=True)
        
        # Create unique filename with timestamp and random suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = os.urandom(3).hex()
        filename = f"{query_type}_{timestamp}_{random_suffix}.parquet"
        filepath = temp_dir / filename
        
        # Save dataframe
        df.to_parquet(str(filepath))
        logger.info(f"Saved dataframe to {filepath}")
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error saving dataframe: {str(e)}")
        return None

def get_temp_files() -> List[str]:
    """
    Get list of temporary data files
    
    Returns:
        list: List of filenames in temp_data directory
    """
    try:
        temp_dir = Path("temp_data")
        if not temp_dir.exists():
            return []
            
        files = [f.name for f in temp_dir.glob("*.parquet")]
        return sorted(files, reverse=True)
        
    except Exception as e:
        logger.error(f"Error getting temp files: {str(e)}")
        return []

def clean_temp_files(keep_current: bool = True) -> None:
    """
    Clean up temporary data files
    
    Args:
        keep_current: Whether to keep the current file
    """
    try:
        temp_dir = Path("temp_data")
        if not temp_dir.exists():
            return
            
        files = list(temp_dir.glob("*.parquet"))
        
        # Sort by modification time
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep most recent file if requested
        if keep_current and files:
            files = files[1:]
            
        # Delete files
        for file in files:
            try:
                file.unlink()
                logger.info(f"Deleted temp file: {file}")
            except Exception as e:
                logger.error(f"Error deleting {file}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error cleaning temp files: {str(e)}")

def get_answer_from_documents(query: str, docs: List[Document], llm) -> str:
    """
    Get an answer from documents using the provided LLM
    
    Args:
        query: User's question
        docs: List of relevant documents
        llm: Language model to use
        
    Returns:
        str: Generated answer
    """
    try:
        # Format context from documents
        context = format_retrieval_context(docs)
        
        # Create prompt
        prompt = create_prompt_with_context(query, context)
        
        # Get answer from LLM
        answer = llm(prompt)
        
        return answer
        
    except Exception as e:
        logger.error(f"Error getting answer from documents: {str(e)}")
        return "Sorry, I was unable to generate an answer from the documents."
