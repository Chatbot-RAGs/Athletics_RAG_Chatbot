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
        
        # Try SQL search for each document
        sql_results = []
        for doc_name in (doc_names or []):
            try:
                # Use table-boosted search if query is table-oriented
                if table_oriented:
                    doc_results = sql_keyword_search(
                        query=query,
                        doc_name=doc_name,
                        include_tables=True,
                        table_boost=5.0,
                        limit=limit
                    )
                else:
                    # Use natural language SQL search for semantic matching
                    doc_results = natural_language_sql_search(
                        query=query,
                        doc_name=doc_name,
                        limit=limit
                    )
                
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
