"""
app_search.py

This module provides advanced search capabilities for the RAG system, including:
1. Natural language SQL search using TF-IDF vectorization for semantic matching
2. Simple SQL search with keyword matching and trigram fuzzy matching
3. SQL keyword search with table-specific boosting
4. Diagnostic functions for troubleshooting vector search issues

Key functions:
- natural_language_sql_search: Uses TF-IDF to find semantically similar content
- simple_sql_search: Fallback search using keywords and fuzzy matching
- sql_keyword_search: Direct keyword-based search with table boosting
- diagnose_vector_search_issues: Troubleshooting tool for vector store problems
"""

import os
import psycopg2
import logging
import json
import re
from langchain_core.documents import Document
from app_database import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_table_structure():
    """
    Check the structure of the documents table
    """
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database")
        return {"error": "Database connection failed"}
    
    try:
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'documents'
            ORDER BY ordinal_position;
        """)
        
        columns = cursor.fetchall()
        logging.info("Table Structure:")
        for col in columns:
            logging.info(f"Column: {col[0]}, Type: {col[1]}")
        
        # Get sample row to verify structure
        cursor.execute("SELECT * FROM documents LIMIT 1")
        
        # Get column names
        col_names = [desc[0] for desc in cursor.description]
        sample = cursor.fetchone()
        
        # Close connection
        cursor.close()
        conn.close()
        
        if sample:
            result = {"columns": col_names, "sample": {}}
            for i, col_name in enumerate(col_names):
                # Truncate content to avoid huge output
                if col_name == 'content' and sample[i]:
                    result["sample"][col_name] = sample[i][:100] + "..." if len(sample[i]) > 100 else sample[i]
                else:
                    result["sample"][col_name] = sample[i]
            return result
        else:
            return {"columns": col_names, "sample": None}
    
    except Exception as e:
        logging.error(f"Error inspecting table: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return {"error": str(e)}

def sql_keyword_search(query, doc_name=None, include_tables=True, table_boost=2.0, limit=50):
    """
    Perform a SQL-based keyword search.
    
    Args:
        query (str): Search query
        doc_name (str, optional): Filter by document name
        include_tables (bool): Whether to specially handle table data
        table_boost (float): Boost factor for results containing tables
        limit (int): Maximum number of results to return
        
    Returns:
        List[Document]: List of matching documents
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        
        # Create a safe version of the query for SQL ILIKE
        safe_query = query.replace("%", "\\%").replace("_", "\\_")
        terms = [term.strip() for term in safe_query.split() if term.strip()]
        
        # If no valid terms, return empty list
        if not terms:
            return []
        
        # Further simplified query structure for robustness
        sql = """
            SELECT id, doc_name, page_number, content, metadata
            FROM documents
            WHERE (
        """
        
        params = []
        conditions = []
        
        # Build search conditions for each term (AND logic between terms)
        for term in terms:
            term_conditions = []
            
            # Basic content search
            term_conditions.append("content ILIKE %s")
            params.append(f"%{term}%")
            
            # Table-specific search if enabled
            if include_tables:
                # Search for terms in content containing table markers
                term_conditions.append("(content ILIKE %s AND content ILIKE %s)")
                params.append(f"%{term}%")
                params.append("%[TABLE%")
                
                # Search using table metadata
                term_conditions.append("(metadata->>'has_tables' = 'true' AND content ILIKE %s)")
                params.append(f"%{term}%")
            
            conditions.append("(" + " OR ".join(term_conditions) + ")")
        
        # Combine all term conditions with AND
        sql += " AND ".join(conditions)
        
        # Add document filter if specified
        if doc_name:
            sql += " AND (doc_name = %s OR metadata->>'source' = %s)"
            params.append(doc_name)
            params.append(doc_name)
            
        # Very simple ORDER BY clause to avoid any issues
        sql += """
            )
            ORDER BY id ASC
            LIMIT %s
        """
        
        # Add limit parameter
        params.append(limit)
        
        # Log the query for debugging (with sensitive values replaced)
        logging.info(f"SQL QUERY: {sql.replace('%s', '?')}")
        logging.info(f"Number of parameters: {len(params)}")
        
        # Execute the query
        cursor.execute(sql, params)
        logging.info("SQL query executed successfully")
        
        # Debug column names
        column_names = [desc[0] for desc in cursor.description]
        logging.info(f"Column names in result: {column_names}")
        
        rows = cursor.fetchall()
        
        logging.info(f"SQL keyword search found {len(rows)} results")
        
        # Process results
        documents = []
        for i, row in enumerate(rows):
            try:
                # Debug row structure
                logging.info(f"Row {i} has {len(row)} items: {[type(item) for item in row]}")
                
                if len(row) < 5:
                    logging.error(f"Row {i} doesn't have enough columns: {row}")
                    continue
                
                doc_id, doc_name, page_num, content, meta = row
                
                # Get metadata
                if meta:
                    # Convert metadata from JSON
                    try:
                        metadata = meta
                        if isinstance(metadata, str) and metadata.strip():
                            try:
                                metadata = json.loads(metadata)
                            except json.JSONDecodeError as je:
                                logging.error(f"JSON decode error: {str(je)} in metadata: {metadata[:100]}")
                                metadata = {"source": doc_name}
                        elif not isinstance(metadata, dict):
                            metadata = {"source": doc_name}
                    except Exception as json_error:
                        logging.error(f"Error parsing metadata JSON: {str(json_error)}")
                        metadata = {"source": doc_name}
                        if page_num:
                            metadata["page"] = page_num
                else:
                    metadata = {"source": doc_name}
                    if page_num:
                        metadata["page"] = page_num
                
                # Check if this chunk contains tables (safely)
                has_tables = False
                if isinstance(metadata, dict):
                    has_tables = metadata.get('has_tables', False)
                if has_tables:
                    logging.info(f"Found chunk with tables: {doc_id}")
                    # Add a flag that can help the LLM identify table content
                    metadata["contains_tables"] = True
                    metadata["result_from_table_search"] = include_tables
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
            except Exception as row_error:
                logging.error(f"Error processing search result row {i}: {str(row_error)}", exc_info=True)
                continue
        
        cursor.close()
        conn.close()
        
        return documents
        
    except Exception as e:
        logging.error(f"SQL keyword search error: {str(e)}", exc_info=True)
        if conn and not conn.closed:
            conn.close()
        return []

def natural_language_sql_search(query: str, doc_name: str, limit: int = 30) -> list:
    """
    Performs a more intelligent SQL search using TF-IDF vectorization to find
    semantically similar content when vector search fails.
    
    Args:
        query: User query in natural language
        doc_name: Document name to filter by
        limit: Maximum number of results
        
    Returns:
        List of matching documents
    """
    # Check for required scikit-learn packages first with a clear error message
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        logger.error(f"Required ML packages missing: {str(e)}.")
        logger.error("Please install scikit-learn and numpy: pip install scikit-learn numpy")
        logger.warning("Falling back to simple SQL search due to missing dependencies")
        return simple_sql_search(query, doc_name, limit)
    
    try:
        # First, get all document chunks for this document using SQL
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to connect to database")
            return []
            
        cursor = conn.cursor()
        
        # Get document chunks
        doc_sql = """
            SELECT id, doc_name, page_number, content, metadata 
            FROM documents 
            WHERE doc_name = %s
            LIMIT 200;
        """
        
        cursor.execute(doc_sql, (doc_name,))
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning(f"No documents found for {doc_name}")
            return []
            
        logger.info(f"Found {len(rows)} document chunks for {doc_name}")
        
        # Extract contents for TF-IDF
        contents = []
        doc_objects = []
        
        for row in rows:
            try:
                # Make sure we have enough columns
                if len(row) < 5:
                    logger.warning(f"Row has fewer than 5 columns: {len(row)}")
                    continue
                
                # Extract data with safety checks
                doc_id = row[0] if row[0] is not None else 0
                row_doc_name = row[1] if row[1] is not None else doc_name
                page_num = row[2] if row[2] is not None else 0
                content = row[3] if row[3] is not None else ""
                meta = row[4]  # Can be None
                
                # Skip empty content
                if not content or len(content.strip()) == 0:
                    continue
                    
                # Add to contents list for vectorization
                contents.append(content)
                
                # Build document object - safely create metadata
                metadata = {"source": row_doc_name}
                if page_num is not None:
                    metadata["page"] = page_num
                
                # Process metadata if available
                if meta:
                    try:
                        if isinstance(meta, str) and meta.strip():
                            # Try to parse JSON metadata
                            meta_dict = json.loads(meta)
                            metadata.update(meta_dict)
                        elif isinstance(meta, dict):
                            # It's already a dict, use it
                            metadata.update(meta)
                    except Exception as meta_error:
                        logger.error(f"Error parsing metadata: {str(meta_error)}")
                    
                doc_objects.append({
                    "content": content,
                    "metadata": metadata
                })
            except Exception as e:
                logger.error(f"Error processing row: {str(e)}", exc_info=True)
                continue
                
        # If we couldn't get any content, return empty list
        if not contents:
            logger.warning("No valid content found in document chunks")
            cursor.close()
            conn.close()
            return []
        
        # Use TF-IDF to find similarity between query and documents
        try:
            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer(stop_words='english')
            
            # Fit and transform document contents
            tfidf_matrix = vectorizer.fit_transform(contents)
            
            # Transform query
            query_vector = vectorizer.transform([query])
            
            # Calculate cosine similarity
            cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            # Get top matches
            top_indices = cosine_similarities.argsort()[-limit:][::-1]
            
            # Filter for minimum similarity
            min_similarity = 0.01  # Small threshold to include more results
            relevant_indices = [idx for idx in top_indices if cosine_similarities[idx] > min_similarity]
            
            logger.info(f"TF-IDF found {len(relevant_indices)} relevant chunks with similarity > {min_similarity}")
            
            # Create Document objects from relevant chunks
            documents = []
            for idx in relevant_indices:
                doc_info = doc_objects[idx]
                documents.append(Document(
                    page_content=doc_info["content"],
                    metadata=doc_info["metadata"]
                ))
            
            cursor.close()
            conn.close()
            return documents
            
        except Exception as e:
            logger.error(f"Error in TF-IDF similarity calculation: {str(e)}")
            # Fall back to simple SQL search
            cursor.close()
            conn.close()
            return simple_sql_search(query, doc_name, limit)
        
    except Exception as e:
        logger.error(f"Error in natural language SQL search: {str(e)}")
        # Fall back to simple SQL search
        return simple_sql_search(query, doc_name, limit)

def simple_sql_search(query: str, doc_name: str, limit: int = 30) -> list:
    """
    Performs a simple SQL search for documents containing keywords from the query.
    This is a fallback for when vector search fails.
    
    Args:
        query: User query
        doc_name: Document name to filter by
        limit: Maximum number of results
        
    Returns:
        List of matching documents
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return []
        
    try:
        cursor = conn.cursor()
        
        # Extract keywords from query (words longer than 2 characters)
        # Using shorter words to increase chances of finding matches
        keywords = [word.lower() for word in query.split() if len(word) > 2]
        logger.info(f"SQL search keywords (length > 2): {keywords}")
        
        # Also try with longer keywords for more precision
        longer_keywords = [word.lower() for word in query.split() if len(word) > 3]
        logger.info(f"SQL search longer keywords (length > 3): {longer_keywords}")
        
        # Generate trigrams from query for more fuzzy matching
        def generate_trigrams(text):
            words = text.lower().split()
            trigrams = []
            for word in words:
                if len(word) > 5:  # Only for longer words
                    for i in range(len(word) - 2):
                        trigrams.append(word[i:i+3])
            return trigrams
            
        trigrams = generate_trigrams(query)
        logger.info(f"Generated trigrams for fuzzy matching: {trigrams}")
        
        if not keywords and not trigrams:
            # If no keywords found, return documents from the specified doc
            sql = """
                SELECT id, doc_name, page_number, content, metadata 
                FROM documents 
                WHERE doc_name = %s
                LIMIT %s;
            """
            cursor.execute(sql, (doc_name, limit))
        else:
            # Try to find exact matches first
            exact_conditions = []
            
            # For exact phrase match
            if len(query.split()) > 1:
                # Try the whole query as a phrase
                exact_conditions.append(f"LOWER(content) LIKE '%{query.lower()}%'")
            
            # Build query conditions for each keyword
            keyword_conditions = []
            for keyword in keywords:
                # More precise match with word boundaries where possible
                if len(keyword) > 4:
                    keyword_conditions.append(f"LOWER(content) ~ '\\m{keyword}\\M'")
                else:
                    keyword_conditions.append(f"LOWER(content) LIKE '%{keyword}%'")
            
            # Add trigram conditions for fuzzy matching
            trigram_conditions = []
            for trigram in trigrams:
                trigram_conditions.append(f"LOWER(content) LIKE '%{trigram}%'")
            
            # Build where clause with weighted prioritization
            where_parts = []
            
            # Exact matches get highest priority
            if exact_conditions:
                where_parts.append(f"({' OR '.join(exact_conditions)})")
            
            # Keyword matches get medium priority
            if keyword_conditions:
                where_parts.append(f"({' OR '.join(keyword_conditions)})")
                
            # Trigram matches get lowest priority
            if trigram_conditions:
                where_parts.append(f"({' OR '.join(trigram_conditions)})")
                
            # Combine with OR between the groups
            combined_where = " OR ".join(where_parts)
            
            # Execute query with document filter
            sql = f"""
                SELECT id, doc_name, page_number, content, metadata 
                FROM documents 
                WHERE doc_name = %s AND ({combined_where})
                LIMIT %s;
            """
            
            logger.info(f"SQL where clause: {combined_where}")
            cursor.execute(sql, (doc_name, limit))
            
        # Process results
        rows = cursor.fetchall()
        logger.info(f"SQL search found {len(rows)} results")
        
        documents = []
        for row in rows:
            try:
                # Add robust error handling for row unpacking
                if row is None or len(row) < 4:
                    logger.warning(f"Invalid row format: {row}")
                    continue
                    
                # Use indexed access instead of unpacking to handle potential issues
                doc_id = row[0] if len(row) > 0 else None
                row_doc_name = row[1] if len(row) > 1 else doc_name
                page_num = row[2] if len(row) > 2 else None
                content = row[3] if len(row) > 3 else ""
                
                # Handle metadata carefully
                if len(row) > 4:
                    meta = row[4]
                else:
                    meta = None
                
                # Skip empty content
                if not content or len(content.strip()) == 0:
                    continue
                
                # Build metadata safely
                metadata = {"source": row_doc_name}
                if page_num is not None:
                    metadata["page"] = page_num
                
                # Handle metadata if available
                if meta:
                    try:
                        if isinstance(meta, str) and meta.strip():
                            try:
                                meta_dict = json.loads(meta)
                                metadata.update(meta_dict)
                            except json.JSONDecodeError:
                                # Not valid JSON, ignore
                                pass
                        elif isinstance(meta, dict):
                            metadata.update(meta)
                    except Exception as meta_error:
                        logger.error(f"Error processing metadata: {str(meta_error)}")
                
                # Create document
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            except Exception as e:
                logger.error(f"Error processing row in SQL search: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Returning {len(documents)} documents from SQL search")
        cursor.close()
        conn.close()
        return documents
        
    except Exception as e:
        logger.error(f"Error in SQL search: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return []

def diagnose_vector_search_issues(doc_name=None, limit=5):
    """
    Diagnostic function to examine documents in the database and verify embedding presence
    
    Args:
        doc_name (str): Optional document name to filter by
        limit (int): Maximum number of records to examine
        
    Returns:
        dict: Diagnostic information
    """
    logging.info("Running vector search diagnostics")
    result = {
        "total_documents": 0,
        "documents_with_embeddings": 0,
        "document_names": [],
        "embedding_samples": [],
        "issues": []
    }
    
    # Connect to database
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database for diagnostics")
        result["issues"].append("Database connection failed")
        return result
    
    try:
        cursor = conn.cursor()
        
        # Check total document count
        cursor.execute("SELECT COUNT(*) FROM documents")
        result["total_documents"] = cursor.fetchone()[0]
        
        # Get list of document names
        cursor.execute("SELECT DISTINCT doc_name FROM documents")
        result["document_names"] = [row[0] for row in cursor.fetchall()]
        
        # Check if langchain_pg_embedding table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'langchain_pg_embedding'
            )
        """)
        embedding_table_exists = cursor.fetchone()[0]
        
        if not embedding_table_exists:
            result["issues"].append("The langchain_pg_embedding table doesn't exist")
            return result
        
        # Count documents with embeddings
        cursor.execute("""
            SELECT COUNT(*) 
            FROM langchain_pg_embedding
        """)
        result["documents_with_embeddings"] = cursor.fetchone()[0]
        
        if result["documents_with_embeddings"] == 0:
            result["issues"].append("No documents have embeddings")
            return result
        
        # Check for specific document if provided
        if doc_name:
            # Try to find the document in the documents table
            cursor.execute("SELECT COUNT(*) FROM documents WHERE doc_name = %s", (doc_name,))
            doc_count = cursor.fetchone()[0]
            
            if doc_count == 0:
                result["issues"].append(f"Document '{doc_name}' not found in documents table")
            else:
                result["doc_count"] = doc_count
                
                # Check if this document has embeddings
                try:
                    cursor.execute("""
                        SELECT COUNT(*) 
                        FROM langchain_pg_embedding 
                        WHERE cmetadata->>'source' = %s
                    """, (doc_name,))
                    embedding_count = cursor.fetchone()[0]
                    result["doc_embedding_count"] = embedding_count
                    
                    if embedding_count == 0:
                        result["issues"].append(f"Document '{doc_name}' has no embeddings")
                except Exception as e:
                    logging.error(f"Error checking embeddings for '{doc_name}': {str(e)}")
                    result["issues"].append(f"Error checking embeddings: {str(e)}")
        
        # Sample some embeddings to verify they are non-empty
        try:
            if doc_name:
                cursor.execute("""
                    SELECT le.embedding, le.cmetadata, d.content
                    FROM langchain_pg_embedding le
                    JOIN documents d ON le.cmetadata->>'source' = d.doc_name
                    WHERE le.cmetadata->>'source' = %s
                    LIMIT %s
                """, (doc_name, limit))
            else:
                cursor.execute("""
                    SELECT le.embedding, le.cmetadata, d.content
                    FROM langchain_pg_embedding le
                    JOIN documents d ON le.cmetadata->>'source' = d.doc_name
                    LIMIT %s
                """, (limit,))
                
            samples = cursor.fetchall()
            
            for i, sample in enumerate(samples):
                embedding = sample[0]
                metadata = sample[1]
                content = sample[2]
                
                sample_info = {
                    "has_embedding": embedding is not None and len(embedding) > 0,
                    "embedding_dimensions": len(embedding) if embedding else 0,
                    "metadata": metadata,
                    "content_preview": content[:100] + "..." if content and len(content) > 100 else content
                }
                
                result["embedding_samples"].append(sample_info)
                
                # Check for potential issues
                if not sample_info["has_embedding"]:
                    result["issues"].append(f"Sample {i+1} has empty embedding")
                if not metadata or not isinstance(metadata, dict):
                    result["issues"].append(f"Sample {i+1} has invalid metadata")
                if "source" not in metadata:
                    result["issues"].append(f"Sample {i+1} missing 'source' in metadata")
                
        except Exception as e:
            logging.error(f"Error sampling embeddings: {str(e)}")
            result["issues"].append(f"Error sampling embeddings: {str(e)}")
        
        cursor.close()
        conn.close()
        
        # Overall assessment
        if not result["issues"]:
            if result["documents_with_embeddings"] < result["total_documents"]:
                result["issues"].append(f"Only {result['documents_with_embeddings']} out of {result['total_documents']} documents have embeddings")
            else:
                result["status"] = "OK - No issues detected"
        
        return result
        
    except Exception as e:
        logging.error(f"Error running vector search diagnostics: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        result["issues"].append(f"Diagnostic error: {str(e)}")
        return result
