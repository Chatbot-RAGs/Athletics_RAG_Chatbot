"""
app_multi_search.py

This module enables searching across multiple documents simultaneously, combining results
from both vector search and semantic SQL search for comprehensive retrieval.

Key functions:
- multi_document_search: Searches across multiple documents using hybrid retrieval techniques,
  combining vector search with natural language SQL search, and ranking results by relevance.
  Handles table-oriented queries with specialized boosting and tracks detailed metrics.
"""

import logging
from typing import List, Optional
from langchain_core.documents import Document
from app_retrieval import DocumentCollection
from app_search import sql_keyword_search, natural_language_sql_search
from app_ranking import rank_docs_by_relevance, is_table_oriented_query
from app_database import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def multi_document_search(query: str, vector_store, doc_names: Optional[List[str]] = None, limit: int = 50) -> DocumentCollection:
    """
    Search across multiple documents using both vector and SQL search
    
    Args:
        query: Search query
        vector_store: Vector store to search in
        doc_names: Optional list of document names to search in
        limit: Maximum number of results to return
        
    Returns:
        DocumentCollection: Collection of matching documents
    """
    try:
        logger.info(f"Multi-document search for '{query}' in docs: {doc_names}")
        
        # Initialize result collection
        results = DocumentCollection()
        seen_contents = set()
        
        # Check if query is table-oriented
        table_oriented = is_table_oriented_query(query)
        if table_oriented:
            logger.info("Query appears to be table-oriented")
        
        # Try vector search first
        try:
            # Create metadata filter
            filter_dict = None
            if doc_names:
                filter_dict = {
                    "metadata": {
                        "source": {
                            "$in": doc_names
                        }
                    }
                }
            
            # Try MMR search first for diversity
            try:
                vector_results = vector_store.max_marginal_relevance_search(
                    query,
                    k=limit,
                    fetch_k=limit*2,
                    lambda_mult=0.7,  # Higher diversity
                    filter=filter_dict
                )
                
                # Add unique results
                for doc in vector_results:
                    if doc.page_content not in seen_contents:
                        results.append(doc)
                        seen_contents.add(doc.page_content)
                        
                logger.info(f"MMR search found {len(vector_results)} results")
                
            except Exception as e:
                logger.error(f"MMR search failed: {str(e)}")
                # Fall back to similarity search
                try:
                    vector_results = vector_store.similarity_search(
                        query,
                        k=limit,
                        filter=filter_dict
                    )
                    
                    # Add unique results
                    for doc in vector_results:
                        if doc.page_content not in seen_contents:
                            results.append(doc)
                            seen_contents.add(doc.page_content)
                            
                    logger.info(f"Similarity search found {len(vector_results)} results")
                    
                except Exception as e:
                    logger.error(f"Similarity search failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
        
        # Track vector results count
        results.vector_count = len(results)
        
        # Try SQL search for each document (or all documents if doc_names is None)
        sql_results = []
        
        # If doc_names is None (All Documents mode), we need to get all document names
        docs_to_search = doc_names if doc_names else []
        
        # If we have no documents to search (and doc_names is None), get all document names from DB
        if not docs_to_search and doc_names is None:
            try:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT doc_name FROM documents")
                    docs_to_search = [row[0] for row in cursor.fetchall()]
                    cursor.close()
                    conn.close()
                    logger.info(f"Retrieved {len(docs_to_search)} document names from database for SQL search")
            except Exception as e:
                logger.error(f"Error getting document names for SQL search: {str(e)}")
        
        # Perform SQL search on each document
        for doc_name in docs_to_search:
            try:
                logger.info(f"Performing SQL search on document: {doc_name}")
                
                # Try both search methods for better coverage
                doc_results = []
                
                # First try natural language search
                try:
                    # Use natural language SQL search for semantic matching
                    nl_results = natural_language_sql_search(
                        query=query,
                        doc_name=doc_name,
                        limit=limit
                    )
                    
                    if nl_results:
                        logger.info(f"Natural language SQL search found {len(nl_results)} results for {doc_name}")
                        doc_results.extend(nl_results)
                except Exception as nl_error:
                    logger.error(f"Natural language SQL search failed for {doc_name}: {str(nl_error)}")
                
                # Then try keyword search (especially if natural language search found nothing)
                if not doc_results or len(doc_results) < 5:
                    try:
                        # Use table-boosted search if query is table-oriented
                        kw_results = sql_keyword_search(
                            query=query,
                            doc_name=doc_name,
                            include_tables=True,
                            table_boost=5.0 if table_oriented else 2.0,
                            limit=limit
                        )
                        
                        if kw_results:
                            logger.info(f"Keyword SQL search found {len(kw_results)} results for {doc_name}")
                            # Add only unique results
                            existing_contents = {doc.page_content for doc in doc_results}
                            for doc in kw_results:
                                if doc.page_content not in existing_contents:
                                    doc_results.append(doc)
                                    existing_contents.add(doc.page_content)
                    except Exception as kw_error:
                        logger.error(f"Keyword SQL search failed for {doc_name}: {str(kw_error)}")
                
                # Add unique results
                for doc in doc_results:
                    if doc.page_content not in seen_contents:
                        # Add source_type metadata to track where result came from
                        if not doc.metadata:
                            doc.metadata = {}
                        doc.metadata['source_type'] = 'sql'
                        
                        sql_results.append(doc)
                        seen_contents.add(doc.page_content)
                
            except Exception as e:
                logger.error(f"SQL search failed for {doc_name}: {str(e)}")
                # Fall back to keyword search if natural language search fails
                try:
                    doc_results = sql_keyword_search(
                        query=query,
                        doc_name=doc_name,
                        include_tables=table_oriented,
                        table_boost=2.0 if table_oriented else 1.0,
                        limit=limit
                    )
                    
                    # Add unique results
                    for doc in doc_results:
                        if doc.page_content not in seen_contents:
                            # Add source_type metadata
                            if not doc.metadata:
                                doc.metadata = {}
                            doc.metadata['source_type'] = 'sql_fallback'
                            
                            sql_results.append(doc)
                            seen_contents.add(doc.page_content)
                except Exception as fallback_error:
                    logger.error(f"Fallback SQL search also failed for {doc_name}: {str(fallback_error)}")
                continue
        
        # Add SQL results to collection
        for doc in sql_results:
            results.append(doc)
            
        # Add source_type metadata to vector results
        for doc in results[:results.vector_count]:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata['source_type'] = 'vector'
        
        # Track SQL results count
        results.sql_count = len(sql_results)
        
        # Track table results if query was table-oriented
        if table_oriented:
            results.table_count = sum(1 for doc in results if doc.metadata.get('has_tables', False))
        
        logger.info(f"Found total {len(results)} results - Vector: {results.vector_count}, SQL: {results.sql_count}, Tables: {results.table_count}")
        
        # Rank results if we have any
        if len(results) > 0:
            try:
                ranked_docs = rank_docs_by_relevance(results, query)
                logger.info(f"Ranked {len(ranked_docs)} documents")
                
                # Create new collection with ranked docs but preserve metrics
                ranked_collection = DocumentCollection(ranked_docs[:limit])
                ranked_collection.vector_count = results.vector_count
                ranked_collection.sql_count = results.sql_count
                ranked_collection.table_count = results.table_count
                
                return ranked_collection
            except Exception as e:
                logger.error(f"Error ranking results: {str(e)}")
                # Return unranked results
                return DocumentCollection(list(results)[:limit])
        
        return results
        
    except Exception as e:
        logger.error(f"Error in multi-document search: {str(e)}")
        return DocumentCollection()
