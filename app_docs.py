"""
app_docs.py

Helper functions for document management in the RAG system.
"""

import logging
from app_database import get_db_connection

logger = logging.getLogger(__name__)

def get_document_stats():
    """
    Get statistics about documents in the database.
    
    Returns:
        dict: Statistics about documents (count, total chunks, etc.)
    """
    try:
        conn = get_db_connection()
        if not conn:
            return {"error": "Could not connect to database"}
            
        cursor = conn.cursor()
        
        # Get total document count (unique doc_names)
        cursor.execute("SELECT COUNT(DISTINCT doc_name) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        # Get total chunks count
        cursor.execute("SELECT COUNT(*) FROM documents")
        chunk_count = cursor.fetchone()[0]
        
        # Get page count
        cursor.execute("SELECT COUNT(DISTINCT CONCAT(doc_name, page_number)) FROM documents")
        page_count = cursor.fetchone()[0]
        
        # Close connection
        cursor.close()
        conn.close()
        
        return {
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "page_count": page_count,
            "avg_chunks_per_doc": chunk_count / doc_count if doc_count > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        return {"error": str(e)}

def get_document_info(doc_name):
    """
    Get detailed information about a specific document.
    
    Args:
        doc_name (str): Name of the document
        
    Returns:
        dict: Document information (page count, chunks per page, etc.)
    """
    try:
        conn = get_db_connection()
        if not conn:
            return {"error": "Could not connect to database"}
            
        cursor = conn.cursor()
        
        # Get page count for this document
        cursor.execute("SELECT COUNT(DISTINCT page_number) FROM documents WHERE doc_name = %s", (doc_name,))
        page_count = cursor.fetchone()[0]
        
        # Get total chunks for this document
        cursor.execute("SELECT COUNT(*) FROM documents WHERE doc_name = %s", (doc_name,))
        chunk_count = cursor.fetchone()[0]
        
        # Get first few chunks as sample
        cursor.execute("SELECT page_number, content FROM documents WHERE doc_name = %s LIMIT 3", (doc_name,))
        samples = [{"page": row[0], "content": row[1][:200] + "..." if len(row[1]) > 200 else row[1]} 
                  for row in cursor.fetchall()]
        
        # Close connection
        cursor.close()
        conn.close()
        
        return {
            "doc_name": doc_name,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "avg_chunks_per_page": chunk_count / page_count if page_count > 0 else 0,
            "samples": samples
        }
        
    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        return {"error": str(e)} 