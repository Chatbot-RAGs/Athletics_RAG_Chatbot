"""
app_streamlit_helpers.py

This module provides cached helper functions for Streamlit to optimize performance.
It includes caching for expensive operations like database connections, document retrieval,
and UI components.
"""

import streamlit as st
import logging
import time
from typing import List, Dict, Optional, Any
import threading

# Global metadata cache with lock for thread safety
_METADATA_CACHE = {}
_METADATA_CACHE_LOCK = threading.Lock()

# Add a background thread for preloading documents
_DOCUMENT_LIST_THREAD = None
_DOCUMENT_LIST_CACHE = {}
_DOCUMENT_LIST_LOCK = threading.Lock()

@st.cache_resource(ttl=3600*12, show_spinner=False)  # Cache for 12 hours
def cached_get_vector_store():
    """
    Get a cached vector store that persists for 12 hours.
    This prevents expensive vector store initialization on each query.
    
    Returns:
        Vector store instance or None if initialization fails
    """
    from app_vector import get_vector_store
    
    try:
        start_time = time.time()
        vector_store = get_vector_store()
        duration = time.time() - start_time
        
        if vector_store:
            logging.info(f"Retrieved cached vector store in {duration:.2f}s")
            return vector_store
        else:
            logging.error("Failed to create cached vector store")
            return None
    except Exception as e:
        logging.error(f"Error in cached vector store: {str(e)}")
        return None

@st.cache_resource(ttl=3600, show_spinner=False)
def get_cached_db_connection():
    """
    Get a cached API session for PostgreSQL MCP server connection that persists for one hour.
    This avoids creating a new session on every rerun.
    
    Returns:
        Session object or None if connection fails
    """
    from app_database import get_api_session
    
    try:
        # Get API session 
        session = get_api_session()
        if session:
            logging.info("Using cached API session for PostgreSQL MCP server")
            return session
        else:
            logging.error("Failed to create cached API session for PostgreSQL MCP server")
            return None
    except Exception as e:
        logging.error(f"Error in cached API session: {str(e)}")
        return None

def preload_document_list():
    """Background thread to preload document list"""
    global _DOCUMENT_LIST_CACHE
    
    try:
        from app_database import load_documents_from_database
        
        # Load documents
        doc_info = load_documents_from_database()
        
        # Update cache with thread safety
        with _DOCUMENT_LIST_LOCK:
            _DOCUMENT_LIST_CACHE = {
                'timestamp': time.time(),
                'data': doc_info
            }
            
        logging.info(f"Preloaded {len(doc_info)} documents in background thread")
    except Exception as e:
        logging.error(f"Error in document preload thread: {str(e)}")

def start_document_preload():
    """Start a background thread to preload document list"""
    global _DOCUMENT_LIST_THREAD
    
    if _DOCUMENT_LIST_THREAD and _DOCUMENT_LIST_THREAD.is_alive():
        return  # Thread already running
        
    _DOCUMENT_LIST_THREAD = threading.Thread(target=preload_document_list)
    _DOCUMENT_LIST_THREAD.daemon = True  # Allow thread to exit when main thread exits
    _DOCUMENT_LIST_THREAD.start()
    
    logging.info("Started document preload thread")

# Modify get_document_list to use the preloaded cache with longer TTL
@st.cache_data(ttl=1800, show_spinner=False)  # Cache for 30 minutes
def get_document_list():
    """
    Get a cached list of documents from the database.
    Refreshes every 30 minutes and uses background preloading.
    
    Returns:
        dict: Document info dictionary with chunk counts
    """
    global _DOCUMENT_LIST_CACHE
    
    try:
        # Check if we have recent preloaded data
        with _DOCUMENT_LIST_LOCK:
            cache = _DOCUMENT_LIST_CACHE
            
        if cache and (time.time() - cache.get('timestamp', 0) < 1800):
            # Use cached data if less than 30 minutes old
            doc_info = cache.get('data', {})
            logging.info(f"Using {len(doc_info)} preloaded documents from cache")
            
            # Start a new preload thread for next time
            start_document_preload()
            return doc_info
            
        # If no valid cache, load documents directly
        from app_database import load_documents_from_database
        
        start_time = time.time()
        doc_info = load_documents_from_database()
        duration = time.time() - start_time
        logging.info(f"Loaded {len(doc_info)} documents from database in {duration:.2f}s")
        
        # Start background preload for next time
        start_document_preload()
        
        return doc_info
    except Exception as e:
        logging.error(f"Error loading cached document list: {str(e)}")
        return {}

# Add a non-Streamlit cache for document metadata to avoid re-fetching
# This is faster than the Streamlit cache for repeated accesses
def get_document_metadata_from_cache(doc_name: str, max_age_seconds: int = 900):
    """
    Get document metadata from the cache if available and not expired.
    
    Args:
        doc_name: Name of the document
        max_age_seconds: Maximum age of the cached data in seconds
    
    Returns:
        dict or None: Document metadata or None if not in cache or expired
    """
    global _METADATA_CACHE
    
    if doc_name not in _METADATA_CACHE:
        return None
        
    timestamp, metadata = _METADATA_CACHE.get(doc_name, (0, None))
    
    # Check if cache is expired
    if time.time() - timestamp > max_age_seconds:
        return None
        
    return metadata

def set_document_metadata_in_cache(doc_name: str, metadata: dict):
    """
    Set document metadata in the cache with the current timestamp.
    
    Args:
        doc_name: Name of the document
        metadata: Document metadata
    
    Returns:
        dict: The metadata that was cached
    """
    global _METADATA_CACHE
    
    with _METADATA_CACHE_LOCK:
        _METADATA_CACHE[doc_name] = (time.time(), metadata)
    
    return metadata

@st.cache_data(ttl=900, show_spinner=False)  # Cache for 15 minutes
def get_document_metadata(doc_name: str):
    """
    Get cached document metadata from the database.
    Uses a fast in-memory cache first, then falls back to the database.
    
    Args:
        doc_name: Name of the document
        
    Returns:
        dict: Document metadata including page count, chunk count, etc.
    """
    from app_documents import get_document_metadata as get_doc_metadata_db
    
    try:
        # First check the fast in-memory cache
        cached_metadata = get_document_metadata_from_cache(doc_name)
        if cached_metadata is not None:
            # If it's in the in-memory cache, don't log to avoid spamming
            return cached_metadata
            
        # If not in memory cache, check database
        start_time = time.time()
        metadata = get_doc_metadata_db(doc_name)
        duration = time.time() - start_time
        
        if metadata and metadata.get("exists", False):
            # Store in the in-memory cache for faster access
            set_document_metadata_in_cache(doc_name, metadata)
            logging.info(f"Retrieved cached metadata for '{doc_name}' in {duration:.2f}s")
            
        return metadata
    except Exception as e:
        logging.error(f"Error getting cached document metadata for '{doc_name}': {str(e)}")
        return {"exists": False, "doc_name": doc_name, "error": str(e)}

@st.cache_data(ttl=60, show_spinner=False)
def cached_hybrid_retriever(query: str, vectorstore, limit: int = 12, doc_name: Optional[str] = None):
    """
    Cached version of hybrid retriever to avoid redundant searches.
    Refreshes every minute.
    
    Args:
        query: The search query
        vectorstore: Vector store to search
        limit: Maximum number of results
        doc_name: Optional document name to filter search
        
    Returns:
        DocumentCollection: Search results
    """
    from app_vector import hybrid_retriever
    
    try:
        start_time = time.time()
        results = hybrid_retriever(query, vectorstore, doc_name=doc_name, max_results=limit)
        duration = time.time() - start_time
        logging.info(f"Retrieved cached search results for '{query}' in {duration:.2f}s")
        return results
    except Exception as e:
        logging.error(f"Error in cached hybrid retriever: {str(e)}")
        from app_documents import DocumentCollection
        return DocumentCollection()  # Return empty results on error

@st.cache_data(ttl=300, show_spinner=False)
def cached_multi_document_search(query: str, vectorstore, doc_names: List[str], limit_per_doc: int = 20):
    """
    Cached version of multi-document search.
    Refreshes every 5 minutes.
    
    Args:
        query: The search query
        vectorstore: Vector store to search
        doc_names: List of document names to search
        limit_per_doc: Maximum results per document
        
    Returns:
        DocumentCollection: Combined search results
    """
    from app_search import multi_document_search
    
    try:
        start_time = time.time()
        results = multi_document_search(query, vectorstore, doc_names, limit_per_doc)
        duration = time.time() - start_time
        logging.info(f"Retrieved cached multi-doc search results for '{query}' in {duration:.2f}s")
        return results
    except Exception as e:
        logging.error(f"Error in cached multi document search: {str(e)}")
        from app_documents import DocumentCollection
        return DocumentCollection()  # Return empty results on error

# UI Helper Functions
def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.1f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"

def display_search_metrics(results):
    """
    Display search metrics in a formatted way
    
    Args:
        results: DocumentCollection with metrics
    """
    if not hasattr(results, 'get_metric'):
        return
        
    metrics = {}
    
    # Collect available metrics
    if hasattr(results, 'get_metric'):
        retrieval_time = results.get_metric('retrieval_time')
        if retrieval_time:
            metrics["Search Time"] = format_time(retrieval_time)
            
        documents_total = results.get_metric('documents_total')
        if documents_total:
            metrics["Documents Searched"] = documents_total
            
        documents_with_results = results.get_metric('documents_with_results')
        if documents_with_results:
            metrics["Documents with Results"] = documents_with_results
            
        vector_count = results.get_metric('vector_count')
        if vector_count:
            metrics["Vector Results"] = vector_count
            
        keyword_count = results.get_metric('keyword_count')
        if keyword_count:
            metrics["Keyword Results"] = keyword_count
    
    # Display metrics in columns
    if metrics:
        cols = st.columns(len(metrics))
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(label, value)

def document_selector(doc_info: Dict, key_prefix: str = "doc_select"):
    """
    Create a document selector component with intelligent defaults
    
    Args:
        doc_info: Dictionary of document information
        key_prefix: Prefix for the component keys
        
    Returns:
        list: Selected document names
    """
    if not doc_info:
        return []
        
    # Get document options
    options = list(doc_info.keys())
    
    # Set default selection
    if len(options) <= 3:
        default = options
    else:
        default = options[:3]  # First 3 documents by default
    
    # Create document selector
    selected = st.multiselect(
        "Select documents to search",
        options=options,
        default=default,
        key=f"{key_prefix}_{len(options)}"
    )
    
    return selected

# Add global cache for document selector default options
# This prevents recalculating defaults on every state change
@st.cache_data(ttl=3600, show_spinner=False)
def get_document_selector_defaults(doc_options):
    """
    Get default document selection options.
    Caches the calculation for an hour.
    
    Args:
        doc_options: List of available document options
        
    Returns:
        list: Default selected documents
    """
    if len(doc_options) <= 3:
        return doc_options
    else:
        return doc_options[:3]  # First 3 documents by default

# Add a placeholder tracking function to ensure placeholders aren't recreated
def get_placeholder(name):
    """
    Get a placeholder by name, creating it if it doesn't exist.
    This prevents recreating placeholders which causes UI flashing.
    
    Args:
        name: Unique name for the placeholder
        
    Returns:
        st.empty: Streamlit placeholder object
    """
    import streamlit as st
    
    placeholder_key = f"placeholder_{name}"
    
    # Return existing placeholder if available
    if placeholder_key in st.session_state:
        return st.session_state[placeholder_key]
    
    # Create new placeholder
    placeholder = st.empty()
    st.session_state[placeholder_key] = placeholder
    return placeholder

# Add a debounce decorator to prevent rapid function calls
def debounce(wait_time):
    """
    Decorator that prevents a function from being called more than once
    in a specified amount of time.
    
    Args:
        wait_time: Number of seconds to wait before allowing another call
    """
    import time
    import functools
    
    def decorator(func):
        # Store the last called time in function attributes
        func._last_call_time = 0
        
        @functools.wraps(func)
        def debounced_function(*args, **kwargs):
            # Get current time
            current_time = time.time()
            
            # Check if enough time has passed since last call
            if current_time - func._last_call_time >= wait_time:
                # Update last call time
                func._last_call_time = current_time
                # Call the original function
                return func(*args, **kwargs)
            # Return None if not enough time has passed
            return None
            
        return debounced_function
    
    return decorator

# Add a memoize decorator to cache function results
def memoize_with_expiry(ttl_seconds=60):
    """
    Decorator that caches function results based on arguments
    with a time-to-live expiration.
    
    Args:
        ttl_seconds: Time to live in seconds for cached results
    """
    import time
    import functools
    
    cache = {}
    
    def decorator(func):
        @functools.wraps(func)
        def memoized_function(*args, **kwargs):
            # Create a key based on function name and arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = "_".join(key_parts)
            
            # Check if result is cached and still valid
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = (result, time.time())
            
            # Clean up expired cache entries
            for k in list(cache.keys()):
                if time.time() - cache[k][1] > ttl_seconds:
                    del cache[k]
                    
            return result
            
        return memoized_function
    
    return decorator

# Add a function to calculate data frame statistics efficiently
@st.cache_data(ttl=300)
def get_dataframe_stats(dataframe):
    """
    Calculate basic statistics for a dataframe efficiently.
    Returns cached results to avoid recalculation.
    
    Args:
        dataframe: Pandas DataFrame
        
    Returns:
        dict: Dictionary of statistics
    """
    import pandas as pd
    
    if dataframe is None or len(dataframe) == 0:
        return None
    
    try:
        stats = {
            'rows': len(dataframe),
            'columns': len(dataframe.columns),
            'column_types': dict(dataframe.dtypes.astype(str)),
            'memory_usage': dataframe.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        }
        
        # Calculate basic stats for numeric columns only
        numeric_cols = dataframe.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_stats = dataframe[numeric_cols].describe().to_dict()
            stats['numeric_stats'] = numeric_stats
            
        # Get sample values for each column (first 5)
        sample_values = {}
        for col in dataframe.columns:
            sample_values[col] = dataframe[col].head(5).tolist()
        stats['sample_values'] = sample_values
        
        return stats
    except Exception as e:
        import logging
        logging.error(f"Error calculating dataframe stats: {str(e)}")
        return None

# Add a function to create cached document preview
@st.cache_data(ttl=600)
def get_document_preview(doc_name, max_chunks=3):
    """
    Get a preview of a document's content for display.
    Uses caching to avoid repeated database queries.
    
    Args:
        doc_name: Name of the document
        max_chunks: Maximum number of chunks to return
        
    Returns:
        list: List of document chunks
    """
    import logging
    from app_database import execute_query
    
    try:
        query = """
            SELECT content, metadata->>'page' as page
            FROM documents
            WHERE doc_name = %s
            ORDER BY page_number ASC
            LIMIT %s
        """
        
        results = execute_query(query, [doc_name, max_chunks])
        chunks = [(row[0], row[1]) for row in results]
        
        return chunks
    except Exception as e:
        logging.error(f"Error getting document preview: {str(e)}")
        return []

# Add a utility function to monitor and report connection pool status
@st.cache_data(ttl=5, show_spinner=False)
def get_pool_status():
    """
    Get current API session status with 5 second TTL
    Using the PostgreSQL MCP server API instead of direct connection pools
    
    Returns:
        dict: API session status information
    """
    from app_database import get_api_session
    
    try:
        # With API-based approach, we don't have direct access to connection pool stats
        # Instead, we can check if the API session is active
        session = get_api_session()
        
        if session:
            return {
                "status": "healthy",
                "message": "PostgreSQL MCP server API session is active",
                "stats": {
                    "used": "N/A",
                    "max_size": "N/A",
                    "used_connections": 0,  # For backward compatibility
                    "total_connections": 0  # For backward compatibility
                },
                "usage_percent": 0
            }
        else:
            return {
                "status": "error",
                "message": "PostgreSQL MCP server API session is not active",
                "stats": {
                    "used": 0,
                    "max_size": 0
                },
                "usage_percent": 0
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting API session status: {str(e)}"
        }

# Add a function to optimize the database pool when needed
def ensure_pool_health(threshold=75):
    """
    Ensure PostgreSQL MCP server API session is healthy
    
    Args:
        threshold: Percentage threshold (unused but kept for backward compatibility)
        
    Returns:
        bool: True if API session is healthy, False otherwise
    """
    from app_database import get_api_session, close_api_session
    
    try:
        # Check API session status
        status = get_pool_status()
        if status.get("status") == "error":
            # Force recreation of API session
            close_api_session()
            new_session = get_api_session()
            return new_session is not None
            
        return True
    except Exception as e:
        logging.error(f"Error ensuring API session health: {str(e)}")
        return False 