"""
app_retrieval.py

This module implements advanced RAG techniques based on the Parent Document Retriever pattern.
It creates parent documents (large chunks) and child documents (small chunks) to maintain 
context while enabling precise retrieval.

Key components:
- DocumentCollection: Extended list class for tracking search metrics
- implement_parent_document_retriever: Sets up the parent-child document structure
- fetch_parent_context: Retrieves parent documents for context enrichment
- hybrid_retriever: Combines vector search and semantic SQL search for comprehensive results

The hybrid retrieval approach ensures high recall by using multiple search strategies:
1. Vector search with MMR for semantic similarity and diversity
2. Natural language SQL search using TF-IDF for semantic matching
3. Table-oriented search with boosting for structured data
4. Document ranking by relevance for optimal result ordering
"""

import logging
from typing import List
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app_embeddings import embeddings
from app_database import get_db_connection, get_connection_string
from app_vector import CustomPGVector
from app_ranking import rank_docs_by_relevance, is_table_oriented_query
from app_search import sql_keyword_search, natural_language_sql_search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for chunk sizes
PARENT_CHUNK_SIZE = 2000  # Larger chunks for context
CHILD_CHUNK_SIZE = 400    # Smaller chunks for precise retrieval
CHUNK_OVERLAP = 50        # Overlap to maintain context between chunks

# Custom document collection class with metrics tracking
class DocumentCollection(list):
    """Extended list class for Document objects with metrics tracking"""
    
    def __init__(self, docs=None):
        super().__init__(docs or [])
        self.sql_count = 0
        self.vector_count = 0
        self.table_count = 0
        self.fallback_count = 0
        self.metrics = {}
    
    def get_metrics(self):
        """Get standardized metrics dictionary"""
        return {
            "sql_results": self.sql_count,
            "vector_results": self.vector_count, 
            "table_results": self.table_count,
            "fallback_results": self.fallback_count,
            "total_results": len(self),
            **self.metrics
        }
    
    def set_metric(self, key, value):
        """Set a custom metric"""
        self.metrics[key] = value
        return self

def implement_parent_document_retriever(documents: List[Document], doc_name: str) -> bool:
    """
    Implements the Parent Document Retriever pattern for the given documents.
    
    Args:
        documents: List of documents to process
        doc_name: Name of the document being processed
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Implementing Parent Document Retriever for {doc_name}")
        
        # Create parent and child splitters with appropriate chunk sizes
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Storage layer for parent documents
        store = InMemoryStore()
        
        # Create vector store for the child documents
        connection_string = get_connection_string()
        if not connection_string:
            logger.error("Failed to get database connection string")
            return False
            
        # Create vector store 
        vectorstore = CustomPGVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name="documents",
            use_jsonb=True
        )
        
        # Create the parent document retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
        
        # Add documents to the retriever
        logger.info(f"Adding {len(documents)} documents to the retriever")
        retriever.add_documents(documents)
        
        logger.info(f"Successfully implemented Parent Document Retriever for {doc_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error implementing Parent Document Retriever: {str(e)}")
        return False

def fetch_parent_context(child_docs, parent_limit=3):
    """
    Fetch parent documents for each chunk to provide more context to the LLM
    
    Args:
        child_docs (DocumentCollection): The document chunks that were retrieved
        parent_limit (int): Max number of parent docs to retrieve
    
    Returns:
        DocumentCollection: Combined child and parent documents
    """
    if not child_docs or len(child_docs) == 0:
        logger.warning("No child documents provided to fetch_parent_context")
        return DocumentCollection()
    
    # Create result collection starting with the child docs
    result_docs = DocumentCollection(list(child_docs))
    
    # Get DB connection
    conn = None
    try:
        conn = get_db_connection()
    except Exception as e:
        logger.error(f"Error connecting to database in fetch_parent_context: {str(e)}")
        return result_docs  # Return child docs if can't connect
    
    if not conn:
        logger.error("Failed to get DB connection in fetch_parent_context")
        return result_docs  # Return child docs if no connection
    
    # Process each child document
    seen_parents = set()
    try:
        cursor = conn.cursor()
        
        for doc in child_docs:
            doc_name = doc.metadata.get("source")
            page_num = doc.metadata.get("page")
            
            if not doc_name or not page_num:
                logger.warning(f"Missing metadata in document: {doc.metadata}")
                continue
                
            # Find parent pages based on proximity
            # Prefer the parent in the immediate vicinity (+/- 1-2 pages)
            parent_nums = []
            curr_page = int(page_num)
            
            # Create a series of ranges, focusing on proximity
            # First try exact -1, +1 pages
            ranges = [
                (curr_page - 1, curr_page - 1),  # Previous page
                (curr_page + 1, curr_page + 1),  # Next page
            ]
            
            # Then try pages further away if needed
            if parent_limit > 2:
                ranges.extend([
                    (curr_page - 3, curr_page - 2),  # Previous 2-3 pages
                    (curr_page + 2, curr_page + 3),  # Next 2-3 pages
                ])
            
            # Execute queries for each range until we have enough parents
            for page_min, page_max in ranges:
                if page_min <= 0:
                    continue  # Skip invalid page numbers
                    
                parent_query = """
                    SELECT id, doc_name, page_number, content
                    FROM documents
                    WHERE doc_name = %s
                    AND page_number BETWEEN %s AND %s
                    AND page_number != %s
                    ORDER BY ABS(page_number - %s)
                    LIMIT %s;
                """
                
                cursor.execute(parent_query, (doc_name, page_min, page_max, 
                                             curr_page, curr_page, parent_limit))
                parents = cursor.fetchall()
                
                # Process found parents
                for parent in parents:
                    if len(parent) < 4:
                        continue
                        
                    parent_id = parent[0]
                    parent_doc = parent[1]
                    parent_page = parent[2]
                    parent_content = parent[3]
                    
                    # Create a unique key for this parent
                    parent_key = f"{parent_doc}_{parent_page}"
                    
                    # Skip if we've already seen this parent
                    if parent_key in seen_parents:
                        continue
                        
                    # Add this parent to results
                    parent_doc_obj = Document(
                        page_content=parent_content,
                        metadata={
                            "source": parent_doc, 
                            "page": parent_page,
                            "is_parent": True,  # Mark as parent
                            "parent_of_page": curr_page
                        }
                    )
                    
                    result_docs.append(parent_doc_obj)
                    seen_parents.add(parent_key)
                    
                # Stop querying if we have enough parents
                if len(seen_parents) >= parent_limit:
                    break
    
    except Exception as e:
        logger.error(f"Error fetching parent context: {str(e)}")
    finally:
        if conn:
            conn.close()
    
    # Count how many parent docs were added
    added_parents = len(result_docs) - len(child_docs)
    logger.info(f"Added {added_parents} parent documents for context")
    
    # Preserve metrics from child_docs and add parent_count
    result_docs.sql_count = getattr(child_docs, 'sql_count', 0)
    result_docs.vector_count = getattr(child_docs, 'vector_count', 0)
    result_docs.fallback_count = getattr(child_docs, 'fallback_count', 0)
    result_docs.parent_count = added_parents
    
    return result_docs

def hybrid_retriever(query: str, vector_store, doc_name: str, limit: int = 30) -> DocumentCollection:
    """
    Combines vector search and SQL search for better results
    
    Args:
        query (str): The search query
        vector_store: The vector store to search in
        doc_name (str): Document name to filter by
        limit (int): Maximum number of results
        
    Returns:
        DocumentCollection: Collection of Document objects with search results and metrics
    """
    logger.info(f"Running hybrid retrieval for '{query}' in {doc_name}")
    
    # Validate inputs
    if not query or not query.strip():
        logger.warning("Empty query provided to hybrid_retriever")
        return DocumentCollection()
        
    if not doc_name:
        logger.warning("No document name provided to hybrid_retriever")
        return DocumentCollection()
    
    # Check if query may be table-oriented
    table_oriented = is_table_oriented_query(query)
    if table_oriented:
        logger.info(f"Query appears to be table-oriented: '{query}'")
    
    # Try both search methods in parallel for better results
    sql_results = []
    vector_results = []
    fallback_used = False
    
    # Natural language SQL search for semantically relevant results
    try:
        logger.info("Running natural language SQL search...")
        
        # If query is table-oriented, use table-specific search with boosting
        if table_oriented:
            logger.info("Using table-boosted search for data query...")
            sql_results = sql_keyword_search(
                query, 
                doc_name=doc_name, 
                include_tables=True,
                table_boost=5.0,  # Higher boost for table data
                limit=limit
            )
        else:
            # Regular search for non-table queries
            sql_results = natural_language_sql_search(query, doc_name, limit=limit)
        
        logger.info(f"Natural language SQL search found {len(sql_results)} results")
        
    except Exception as e:
        logger.error(f"SQL search failed: {str(e)}")
    
    # Vector search for semantic similarity
    try:
        logger.info("Running vector search...")
        # Create filter for vector search
        filter_dict = {
            "metadata": {
                "source": {
                    "$eq": doc_name
                }
            }
        }
        
        # Check if vector_store is properly initialized
        if vector_store is None:
            logger.error("Vector store is None - cannot perform vector search")
        else:
            # Try different search strategies
            all_results = []
            seen_content = set()
            
            # Strategy 1: MMR with balanced diversity (0.5)
            try:
                results = vector_store.max_marginal_relevance_search(
                    query,
                    k=20,
                    fetch_k=50,
                    lambda_mult=0.5,
                    filter=filter_dict
                )
                
                # Add unique results
                for doc in results:
                    if doc.page_content not in seen_content:
                        all_results.append(doc)
                        seen_content.add(doc.page_content)
                        
                logger.info(f"MMR (0.5) search found {len(results)} results")
            except Exception as e:
                logger.error(f"Error with MMR (0.5) search: {str(e)}")
            
            # Strategy 2: Pure similarity search
            try:
                results = vector_store.similarity_search(
                    query,
                    k=20,
                    filter=filter_dict
                )
                
                # Add unique results
                for doc in results:
                    if doc.page_content not in seen_content:
                        all_results.append(doc)
                        seen_content.add(doc.page_content)
                        
                logger.info(f"Similarity search found {len(results)} results")
            except Exception as e:
                logger.error(f"Error with similarity search: {str(e)}")
            
            vector_results = all_results
            logger.info(f"Vector search found {len(vector_results)} total results")
            
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
    
    # Combine unique results
    result_collection = DocumentCollection()
    seen_contents = set()
    
    # First add SQL results
    for doc in sql_results:
        if doc.page_content and doc.page_content not in seen_contents:
            result_collection.append(doc)
            seen_contents.add(doc.page_content)
    
    # Then add vector results that aren't duplicates
    for doc in vector_results:
        if doc.page_content and doc.page_content not in seen_contents:
            result_collection.append(doc)
            seen_contents.add(doc.page_content)
    
    # Track metrics
    result_collection.sql_count = len(sql_results)
    result_collection.vector_count = len(vector_results)
    result_collection.table_count = len(sql_results) if table_oriented else 0
    result_collection.fallback_count = len(sql_results) if fallback_used else 0
    
    logger.info(f"Combined results before ranking: {len(result_collection)} chunks")
    logger.info(f"Metrics - SQL: {result_collection.sql_count}, Vector: {result_collection.vector_count}, Tables: {result_collection.table_count}, Fallback: {result_collection.fallback_count}")
    
    # If we have at least some results, rank them
    if len(result_collection) > 0:
        try:
            # Rank documents by keyword relevance
            ranked_docs = rank_docs_by_relevance(result_collection, query)
            logger.info(f"Ranked {len(ranked_docs)} documents by keyword relevance")
            
            # Create a new collection with the ranked docs but preserve metrics
            ranked_collection = DocumentCollection(ranked_docs[:limit])
            ranked_collection.sql_count = result_collection.sql_count
            ranked_collection.vector_count = result_collection.vector_count
            ranked_collection.table_count = result_collection.table_count
            ranked_collection.fallback_count = result_collection.fallback_count
            
            # Log top 3 documents for debugging
            for i, doc in enumerate(ranked_collection[:3] if len(ranked_collection) >= 3 else ranked_collection):
                logger.info(f"Top result {i+1}: {doc.page_content[:100]}...")
            
            # Return the limited results
            return ranked_collection
        except Exception as e:
            logger.error(f"Error during document ranking: {str(e)}")
            # Fall back to unranked results but keep metrics
            limited_collection = result_collection[:limit]
            return limited_collection
    else:
        # No results after combining
        return DocumentCollection()
