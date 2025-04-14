"""
app_database.py

This module contains utilities for database operations.
Handles connections to PostgreSQL database with pgvector extension for vector similarity search.
Provides functions for:
- Creating database connection strings
- Establishing database connections
- Initializing pgvector extension and tables
- Saving document chunks and their embeddings to the database
"""

import os
import psycopg2
from psycopg2.extras import execute_values
import streamlit as st
import logging
import json
import math  # For NaN handling
import threading
import re

# Helper function to make data JSON-safe (handle NaN, Infinity, etc.)
def make_json_safe(obj):
    """
    Convert an object to be JSON-safe, handling NaN, Infinity, and other values that don't serialize well
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    else:
        return obj

def get_connection_string():
    # Get database credentials from environment variables
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    
    # Ensure all parts are present before forming the string
    if all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        logging.info("Database connection string configured.")
        return connection_string
    else:
        logging.error("Database connection details missing in .env file.")
        st.error("Database connection details missing in .env file.")
        return None

def get_db_connection():
    """
    Create a connection to the PostgreSQL database with SSL support.
    
    Returns:
        Connection object or None if connection failed
    """
    try:
        # Get connection parameters with explicit logging
        db_name = os.getenv("POSTGRES_DATABASE", os.getenv("DB_NAME"))
        db_user = os.getenv("POSTGRES_USER", os.getenv("DB_USER")) 
        db_pwd = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD"))
        db_host = os.getenv("POSTGRES_HOST", os.getenv("DB_HOST"))
        db_port = os.getenv("POSTGRES_PORT", os.getenv("DB_PORT"))
        
        # Check if all required parameters are present
        if not all([db_name, db_user, db_pwd, db_host, db_port]):
            missing = []
            if not db_name: missing.append("POSTGRES_DATABASE/DB_NAME")
            if not db_user: missing.append("POSTGRES_USER/DB_USER")
            if not db_pwd: missing.append("POSTGRES_PASSWORD/DB_PASSWORD")
            if not db_host: missing.append("POSTGRES_HOST/DB_HOST")
            if not db_port: missing.append("POSTGRES_PORT/DB_PORT")
            logging.error(f"Missing database connection parameters: {', '.join(missing)}")
            return None
        
        # Build connection parameters with SSL support for Aiven Cloud
        conn_params = {
            "dbname": db_name,
            "user": db_user,
            "password": db_pwd,
            "host": db_host,
            "port": db_port,
            "connect_timeout": 5,
            "sslmode": "require"  # Aiven Cloud requires SSL
        }
        
        # Check if CA certificate is available in env vars or current directory
        pg_ssl_ca = os.getenv('PG_SSL_CA')
        if not pg_ssl_ca or not os.path.exists(pg_ssl_ca):
            if os.path.exists('ca.pem'):
                pg_ssl_ca = 'ca.pem'
                logging.info(f"Using CA certificate from current directory: {pg_ssl_ca}")
                
        if pg_ssl_ca and os.path.exists(pg_ssl_ca):
            conn_params["sslrootcert"] = pg_ssl_ca
            conn_params["sslmode"] = "verify-ca"
            logging.info(f"Using SSL with certificate: {pg_ssl_ca}")
        
        # Log the connection attempt without exposing credentials
        logging.info(f"Connecting to database {conn_params['dbname']} at {conn_params['host']}:{conn_params['port']}")
        
        # Connect with the configured parameters
        conn = psycopg2.connect(**conn_params)
        logging.info("Database connection successful.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {str(e)}")
        return None

def connect_mysql(db_name=None):
    """
    Create a connection to the MySQL database with SSL support.
    
    Args:
        db_name: Optional database name override
        
    Returns:
        Connection object or None if connection failed
    """
    try:
        # Get connection parameters
        mysql_db_name = db_name or os.getenv('MYSQL_DB_NAME', os.getenv('MYSQL_DATABASE', 'mysql'))
        mysql_db_user = os.getenv('MYSQL_DB_USER', os.getenv('MYSQL_USER', 'root'))
        mysql_db_password = os.getenv('MYSQL_DB_PASSWORD', os.getenv('MYSQL_PASSWORD', ''))
        mysql_db_host = os.getenv('MYSQL_DB_HOST', os.getenv('MYSQL_HOST', 'localhost'))
        mysql_db_port = os.getenv('MYSQL_DB_PORT', os.getenv('MYSQL_PORT', '3306'))
        
        # Build connection parameters
        conn_params = {
            "host": mysql_db_host,
            "port": int(mysql_db_port),
            "user": mysql_db_user,
            "connect_timeout": 5,
            "charset": 'utf8mb4'
        }
        
        # Only include password if it's not empty
        if mysql_db_password:
            conn_params["password"] = mysql_db_password
        
        # Always include database name in connection parameters
        conn_params["database"] = mysql_db_name
        
        # Check for SSL CA certificate
        mysql_ssl_ca = os.getenv('MYSQL_SSL_CA')
        if not mysql_ssl_ca or not os.path.exists(mysql_ssl_ca):
            if os.path.exists('ca.pem'):
                mysql_ssl_ca = 'ca.pem'
                logging.info(f"Using CA certificate from current directory: {mysql_ssl_ca}")
                
        if mysql_ssl_ca and os.path.exists(mysql_ssl_ca):
            logging.info(f"Using SSL CA certificate: {mysql_ssl_ca}")
            ssl_args = {
                'ssl': {
                    'ca': mysql_ssl_ca,
                }
            }
            conn_params.update(ssl_args)
            
        # Try connection with avnadmin username if Aiven Cloud is detected
        mysql_is_aiven = mysql_db_host and ("aivencloud" in mysql_db_host.lower() or "aiven" in mysql_db_host.lower())
        conn = None
        
        try:
            # Connect to database with configured parameters
            import pymysql
            conn = pymysql.connect(**conn_params)
        except Exception as e:
            if mysql_is_aiven and mysql_db_user == "anvadmin":
                # Try with the correct username for Aiven (avnadmin)
                logging.info("First connection attempt failed, trying with 'avnadmin' username")
                conn_params["user"] = "avnadmin"
                conn = pymysql.connect(**conn_params)
            else:
                # Re-raise if not an Aiven username issue
                raise
                
        logging.info(f"Connected to MySQL database: {conn_params.get('database', 'mysql')}")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to MySQL: {str(e)}")
        return None

def check_database_status(db_type="postgres"):
    """
    Check if a database connection can be established.
    
    Args:
        db_type: "postgres" or "mysql"
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        if db_type.lower() == "postgres":
            conn = get_db_connection()
        else:
            conn = connect_mysql()
            
        if conn:
            conn.close()
            return True
        return False
    except Exception as e:
        logging.error(f"Error checking {db_type} connection: {str(e)}")
        return False
        
def execute_postgres_query(query, db_name=None):
    """Execute a query on the PostgreSQL database.
    
    Args:
        query (str): SQL query to execute
        db_name (str): Optional database name override
        
    Returns:
        dict: Query results with success flag, data, and optionally a DataFrame
    """
    import pandas as pd
    import logging
    
    logging.info(f"Executing PostgreSQL query: {query}")
    
    # Check if the query contains a potential table name with uppercase letters
    # and modify the query to use quotes if needed
    modified_query = query
    
    # Look for table names in FROM and JOIN clauses
    from_match = re.search(r'\bFROM\s+([a-zA-Z0-9_\.]+)', query, re.IGNORECASE)
    if from_match:
        table_name = from_match.group(1)
        # Handle schema.table format
        if '.' in table_name:
            schema, table = table_name.split('.')
            sanitized_table = sanitize_table_name(table)
            if sanitized_table != table:
                # Replace the table name with the sanitized version
                modified_query = modified_query.replace(table_name, f"{schema}.{sanitized_table}")
        else:
            sanitized_table = sanitize_table_name(table_name)
            if sanitized_table != table_name:
                # Replace the table name with the sanitized version
                modified_query = modified_query.replace(f" {table_name}", f" {sanitized_table}")
    
    # Also check for JOIN clauses
    join_matches = re.finditer(r'\bJOIN\s+([a-zA-Z0-9_\.]+)', query, re.IGNORECASE)
    for match in join_matches:
        table_name = match.group(1)
        # Handle schema.table format
        if '.' in table_name:
            schema, table = table_name.split('.')
            sanitized_table = sanitize_table_name(table)
            if sanitized_table != table:
                # Replace the table name with the sanitized version
                modified_query = modified_query.replace(table_name, f"{schema}.{sanitized_table}")
        else:
            sanitized_table = sanitize_table_name(table_name)
            if sanitized_table != table_name:
                # Replace the table name with the sanitized version
                modified_query = modified_query.replace(f" {table_name}", f" {sanitized_table}")
    
    if modified_query != query:
        logging.info(f"Query modified for case sensitivity: {modified_query}")
    
    try:
        # Override DB name if provided
        original_db_name = os.getenv("DB_NAME")
        if db_name:
            os.environ["DB_NAME"] = db_name
            
        # Connect to database with SSL support
        conn = get_db_connection()
        
        # Restore original DB name
        if db_name:
            os.environ["DB_NAME"] = original_db_name
            
        if not conn:
            return {
                "success": False,
                "error": "Failed to connect to PostgreSQL database",
                "data": []
            }
        
        # Execute query
        cursor = conn.cursor()
        cursor.execute(modified_query)
        
        # Check if query was a SELECT query
        if modified_query.strip().upper().startswith("SELECT"):
            # Get column names
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Get results
            results = cursor.fetchall()
            
            # Convert to DataFrame if there are results
            dataframe = None
            if results:
                try:
                    dataframe = pd.DataFrame(results, columns=column_names)
                except Exception as e:
                    logging.error(f"Error creating DataFrame: {str(e)}")
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            # Return results
            return {
                "success": True,
                "data": results,
                "dataframe": dataframe,
                "column_names": column_names
            }
        else:
            # For non-SELECT queries (INSERT, UPDATE, DELETE)
            conn.commit()
            affected_rows = cursor.rowcount
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            # Return results for non-SELECT query
            return {
                "success": True,
                "data": [],
                "affected_rows": affected_rows,
                "dataframe": pd.DataFrame()
            }
    
    except Exception as e:
        # Log error
        logging.error(f"Error executing PostgreSQL query: {str(e)}")
        
        # Make sure to close connection if it exists
        if 'conn' in locals() and conn:
            conn.close()
        
        # Return error response
        return {
            "success": False,
            "error": str(e),
            "data": []
        }

def execute_mysql_query(query, db_name=None):
    """Execute a MySQL query and return the results
    
    Args:
        query (str): SQL query to execute
        db_name (str): Optional database name to connect to
        
    Returns:
        dict: Query results with success flag, data, and optionally a DataFrame
    """
    import pymysql
    import pandas as pd
    import logging
    
    logging.info(f"Executing MySQL query: {query}")
    
    # Get database configuration from environment variables
    mysql_host = os.getenv("MYSQL_DB_HOST", os.getenv("MYSQL_HOST", "localhost"))
    mysql_port = int(os.getenv("MYSQL_DB_PORT", os.getenv("MYSQL_PORT", "3306")))
    mysql_user = os.getenv("MYSQL_DB_USER", os.getenv("MYSQL_USER", "root"))
    mysql_password = os.getenv("MYSQL_DB_PASSWORD", os.getenv("MYSQL_PASSWORD", ""))
    mysql_database = db_name or os.getenv("MYSQL_DB_NAME", os.getenv("MYSQL_DATABASE", "mysql"))
    
    # Check if CA certificate exists
    ca_cert = "ca.pem"
    if os.path.exists(ca_cert):
        ssl_config = {"ca": ca_cert}
        logging.info(f"Using CA certificate from current directory: {ca_cert}")
    else:
        # Check in parent directory
        parent_ca_cert = os.path.join("..", ca_cert)
        if os.path.exists(parent_ca_cert):
            ssl_config = {"ca": parent_ca_cert}
            logging.info(f"Using CA certificate from parent directory: {parent_ca_cert}")
        else:
            ssl_config = None
            logging.warning("No CA certificate found, not using SSL")
    
    try:
        # Determine appropriate cursor class
        cursorclass = pymysql.cursors.DictCursor  # Default to dictionary cursor
        
        # Connect to the database
        conn = connect_mysql(db_name)
        if not conn:
            return {"success": False, "error": "Failed to connect to MySQL database"}
        
        # Execute the query
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Check if query was a SELECT query
        if query.strip().upper().startswith("SELECT"):
            # Fetch results
            rows = cursor.fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Convert to DataFrame if there are results
            dataframe = None
            if rows:
                try:
                    # Check what kind of cursor we have by examining first row
                    if isinstance(rows[0], dict):
                        # Dict cursor - data already has column names as keys
                        dataframe = pd.DataFrame(rows)
                    else:
                        # Tuple cursor - need to add column names
                        dataframe = pd.DataFrame(rows, columns=column_names)
                except Exception as e:
                    logging.error(f"Error creating DataFrame: {str(e)}")
            
            # Close connections
            cursor.close()
            conn.close()
            
            # Return results
            return {
                "success": True,
                "data": rows,
                "dataframe": dataframe,
                "column_names": column_names
            }
        else:
            # For non-SELECT queries
            conn.commit()
            affected_rows = cursor.rowcount
            
            # Close connections
            cursor.close()
            conn.close()
            
            # Return results for non-SELECT query
            return {
                "success": True,
                "data": [],
                "affected_rows": affected_rows,
                "dataframe": pd.DataFrame()
            }
    
    except Exception as e:
        logging.error(f"Error executing MySQL query: {str(e)}")
        # Make sure to close connection if it exists
        if 'conn' in locals() and conn:
            conn.close()
        
        return {
            "success": False,
            "error": str(e),
            "data": []
        }

def check_pgvector_extension():
    """Check if pgvector extension is installed in the database"""
    conn = None
    try:
        # Get connection
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database to check pgvector")
            return False
        
        # Check if pgvector extension exists
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector';
        """)
        result = cursor.fetchone()[0]
        
        # Check result
        if result > 0:
            logging.info("pgvector extension is installed")
            
            # Also check for langchain_pg_embedding table
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'langchain_pg_embedding'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                logging.info("langchain_pg_embedding table exists")
                return True
            else:
                logging.error("pgvector extension is installed but langchain_pg_embedding table is missing")
                return False
        else:
            logging.error("pgvector extension is NOT installed")
            return False
    except Exception as e:
        logging.error(f"Error checking pgvector extension: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def initialize_pgvector():
    """
    Initialize the database with pgvector extension and required tables.
    More forceful implementation that will attempt to create the extension
    and tables if they don't exist.
    """
    conn = None
    try:
        # First check existing connection
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database for initialization")
            return False
        
        cursor = conn.cursor()
        
        # Step 1: Create pgvector extension if it doesn't exist
        logging.info("Checking for pgvector extension...")
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            logging.info("pgvector extension created or already exists")
        except Exception as e:
            logging.error(f"Failed to create pgvector extension: {str(e)}")
            logging.warning("You may need to install pgvector extension on your PostgreSQL server")
            return False
        
        # Step 2: Check if documents table exists and has proper structure
        logging.info("Setting up documents table...")
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    doc_name TEXT NOT NULL,
                    page_number INTEGER,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB
                );
            """)
            
            # Create index for documents table
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            conn.commit()
            logging.info("Documents table created or verified")
        except Exception as e:
            logging.error(f"Error setting up documents table: {str(e)}")
        
        # Step 3: Create custom_metadata_match function for PGVector
        logging.info("Creating custom metadata match function...")
        try:
            cursor.execute("""
                CREATE OR REPLACE FUNCTION custom_metadata_match(metadata JSONB, field TEXT, value TEXT)
                RETURNS BOOLEAN AS $$
                BEGIN
                    RETURN metadata->>field = value OR metadata->field @> value::jsonb;
                EXCEPTION
                    WHEN others THEN
                        RETURN metadata->>field = value;
                END;
                $$ LANGUAGE plpgsql;
            """)
            conn.commit()
            logging.info("Custom metadata match function created or updated")
        except Exception as e:
            logging.error(f"Error creating custom metadata match function: {str(e)}")
        
        # Step 4: Check if langchain_pg_embedding table exists
        logging.info("Checking for langchain_pg_embedding table...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'langchain_pg_embedding'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        # Step 5: If table doesn't exist, create it manually
        if not table_exists:
            logging.info("Creating langchain_pg_embedding table...")
            try:
                # Create the table with required structure for PGVector
                cursor.execute("""
                    CREATE TABLE langchain_pg_embedding (
                        uuid UUID PRIMARY KEY,
                        cmetadata JSONB,
                        document TEXT,
                        embedding VECTOR(1536)
                    );
                """)
                
                # Create index for faster similarity search
                cursor.execute("""
                    CREATE INDEX ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
                """)
                
                conn.commit()
                logging.info("Successfully created langchain_pg_embedding table and index")
            except Exception as e:
                logging.error(f"Failed to create langchain_pg_embedding table: {str(e)}")
                return False
        else:
            logging.info("langchain_pg_embedding table already exists")
        
        cursor.close()
        conn.close()
        
        # Verify if everything is properly set up
        if check_pgvector_extension():
            logging.info("pgvector setup complete and verified")
            return True
        else:
            logging.error("pgvector setup could not be verified")
            return False
        
    except Exception as e:
        logging.error(f"Error initializing pgvector: {str(e)}")
        if conn and not conn.closed:
            conn.close()
        return False

def save_document_to_db(doc_name, chunks, embeddings, metadata_list):
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cur:
            # Debug logging for document information
            logging.info(f"Saving document to database: {doc_name} with {len(chunks)} chunks")
            
            # Log a sample of the data being stored
            if chunks and len(chunks) > 0:
                logging.info(f"Sample chunk: {chunks[0][:100]}...")
                
            if metadata_list and len(metadata_list) > 0:
                # Check if there are tables in the first metadata item
                if metadata_list[0].get('has_tables', False):
                    logging.info(f"Document contains tables. First page has {metadata_list[0].get('tables_count', 0)} tables.")
                logging.info(f"Sample metadata: {str(metadata_list[0])[:200]}...")
            
            # Ensure metadata has the proper structure for JSONB querying
            # Metadata should have both 'source' and 'page' fields at a minimum
            validated_metadata_list = []
            for metadata in metadata_list:
                # Create a copy to avoid modifying the original metadata
                validated_metadata = dict(metadata)
                
                # Ensure 'source' field exists and equals doc_name
                if 'source' not in validated_metadata:
                    validated_metadata['source'] = doc_name
                
                # Ensure 'page' field exists
                if 'page' not in validated_metadata:
                    validated_metadata['page'] = None
                
                # Handle table metadata - validate and clean tables data
                if 'tables' in validated_metadata and isinstance(validated_metadata['tables'], list):
                    # Ensure tables data is compact but complete by keeping essential fields
                    for table in validated_metadata['tables']:
                        # We can remove the full table_text from metadata as it's already in the content
                        if 'table_text' in table:
                            del table['table_text']
                        
                        # Keep table_data and a compact version of table_html
                        # You could also remove table_html if space is a concern
                
                validated_metadata_list.append(validated_metadata)
                
            # Log first validated metadata for debugging
            if validated_metadata_list:
                logging.info(f"First validated metadata sample: {str(validated_metadata_list[0])[:200]}...")
            
            # Prepare data for batch insert
            data = []
            for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, validated_metadata_list)):
                # Check if this chunk contains tables
                has_tables = metadata.get('has_tables', False)
                
                # Add table-specific tags to the metadata for enhanced searching
                if has_tables:
                    # Add a tag to make searching for tables easier
                    metadata['contains_table'] = True
                    metadata['searchable_table'] = True
                
                # Make metadata JSON-safe
                json_safe_metadata = make_json_safe(metadata)
                
                data.append((
                    doc_name,
                    metadata.get('page', None),
                    chunk,
                    embedding,
                    json.dumps(json_safe_metadata) # Serialize metadata dict to JSON string
                ))

            # Batch insert
            execute_values(
                cur,
                """
                INSERT INTO documents (doc_name, page_number, content, embedding, metadata)
                VALUES %s
                """,
                data
            )

            conn.commit()
            logging.info(f"Successfully saved {len(chunks)} chunks for document: {doc_name}")
        return True
    except Exception as e:
        st.error(f"Error saving to database: {str(e)}")
        logging.error(f"Error saving document to database: {str(e)}", exc_info=True)
        return False
    finally:
        conn.close()

def load_documents_from_database():
    """Load document names directly from the database"""
    try:
        # Connect to database using env variables
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        
        if not all([db_name, db_user, db_password, db_host, db_port]):
            logging.error("Database credentials not found in environment variables")
            return {}
        
        # Connect to the database
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        
        cursor = conn.cursor()
        
        # Get unique document names from documents table
        cursor.execute("""
            SELECT DISTINCT metadata->>'source' as doc_name, COUNT(*) as chunk_count
            FROM documents 
            WHERE metadata->>'source' IS NOT NULL
            GROUP BY metadata->>'source'
            ORDER BY doc_name;
        """)
        docs_from_documents = cursor.fetchall()
        
        # Store documents from documents table
        doc_info = {doc[0]: {"chunk_count": doc[1]} for doc in docs_from_documents}
        
        # Also check langchain_pg_embedding table
        cursor.execute("""
            SELECT DISTINCT cmetadata->>'source' as doc_name, COUNT(*) as chunk_count
            FROM langchain_pg_embedding
            WHERE cmetadata->>'source' IS NOT NULL
            GROUP BY cmetadata->>'source'
            ORDER BY doc_name;
        """)
        docs_from_langchain = cursor.fetchall()
        
        # Merge results from both tables
        for doc in docs_from_langchain:
            doc_name = doc[0]
            if doc_name in doc_info:
                # Update the chunk count if the document exists in both tables
                doc_info[doc_name]["langchain_chunks"] = doc[1]
                doc_info[doc_name]["in_langchain"] = True
            else:
                # Add the document if it's only in langchain_pg_embedding
                doc_info[doc_name] = {
                    "chunk_count": 0,  # Not in documents table
                    "langchain_chunks": doc[1],
                    "in_langchain": True
                }
        
        # Add in_langchain flag for documents that are only in documents table
        for doc_name in doc_info:
            if "in_langchain" not in doc_info[doc_name]:
                doc_info[doc_name]["in_langchain"] = False
                doc_info[doc_name]["langchain_chunks"] = 0
        
        # Close connection
        cursor.close()
        conn.close()
        
        logging.info(f"Loaded {len(doc_info)} documents from database:")
        for doc_name, info in doc_info.items():
            logging.info(f"  - {doc_name}: {info['chunk_count']} chunks in documents table, {info.get('langchain_chunks', 0)} chunks in langchain_pg_embedding table")
        
        return doc_info
    except Exception as e:
        logging.error(f"Error loading documents from database: {str(e)}")
        return {}

def inspect_database_contents():
    """Retrieve and log sample documents from the database for debugging purposes"""
    try:
        # Connect to database using env variables
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database for inspection")
            return
        
        cursor = conn.cursor()
        
        # Check total document count
        cursor.execute("SELECT COUNT(*) FROM documents;")
        count = cursor.fetchone()
        logging.info(f"Total document chunks in database: {count[0]}")
        
        # Check document names and counts
        cursor.execute("SELECT doc_name, COUNT(*) FROM documents GROUP BY doc_name;")
        doc_counts = cursor.fetchall()
        logging.info("Document counts by doc_name:")
        for doc in doc_counts:
            logging.info(f"- {doc[0]}: {doc[1]} chunks")
        
        # Check metadata structure for a few samples
        cursor.execute("SELECT id, doc_name, metadata FROM documents LIMIT 3;")
        sample_rows = cursor.fetchall()
        logging.info("Sample document rows:")
        for row in sample_rows:
            logging.info(f"ID: {row[0]}")
            logging.info(f"doc_name: {row[1]}")
            logging.info(f"metadata: {row[2]}")
            logging.info("---")
            
        # Close connection
        cursor.close()
        conn.close()
        
        return {
            "total_count": count[0] if count else 0,
            "doc_counts": doc_counts,
            "sample_rows": sample_rows
        }
    except Exception as e:
        logging.error(f"Error inspecting database: {str(e)}", exc_info=True)
        return None 

def initialize_database(min_connections=5, max_connections=20):
    """
    Initialize database structures directly using PostgreSQL connection.
    This function sets up necessary database structures for the application.
    
    Args:
        min_connections: Minimum connections in the pool (if pooling is implemented)
        max_connections: Maximum connections in the pool (if pooling is implemented)
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    logging.info("Initializing database structures with direct connection")
    
    try:
        # First use initialize_pgvector to set up the vector extension
        if not initialize_pgvector():
            logging.error("Failed to initialize pgvector extension")
            return False
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database for initialization")
            return False
        
        # Create user_queries table if it doesn't exist
        try:
            cursor = conn.cursor()
            
            # Check if user_queries table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'user_queries'
                )
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logging.info("Creating user_queries table...")
                
                # Create table for storing user queries
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_queries (
                    id SERIAL PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    query_embedding vector(1536),
                    result JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)
                conn.commit()
                logging.info("user_queries table created successfully")
            else:
                logging.info("user_queries table already exists")
            
            cursor.close()
            conn.close()
            
            logging.info("Database initialization completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error creating user_queries table: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            if conn and not conn.closed:
                conn.close()
            return False
        
    except Exception as e:
        logging.error(f"Error in database initialization: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False 

def delete_document(doc_name):
    """
    Delete a document from the database
    
    Args:
        doc_name: Document name to delete
        
    Returns:
        int: Number of chunks deleted
    """
    logging.info(f"Deleting document {doc_name}")
    
    try:
        return delete_document_from_database(doc_name)
    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        return 0

def delete_document_from_database(doc_name):
    """Delete document from database"""
    conn = None
    try:
        logging.info(f"Deleting document {doc_name} from database")
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database for document deletion")
            return 0
            
        cursor = conn.cursor()
        
        # Get document count before deleting
        cursor.execute("""
            SELECT COUNT(*) 
            FROM documents 
            WHERE doc_name = %s OR metadata->>'source' = %s
        """, (doc_name, doc_name))
        chunk_count = cursor.fetchone()[0]
        
        if chunk_count == 0:
            logging.info(f"No chunks found for document {doc_name}")
            cursor.close()
            conn.close()
            return 0
            
        # Delete from documents table
        cursor.execute("""
            DELETE FROM documents 
            WHERE doc_name = %s OR metadata->>'source' = %s
            RETURNING id;
        """, (doc_name, doc_name))
        
        # Also try to delete from langchain_pg_embedding if it exists
        try:
            # Check if langchain_pg_embedding table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'langchain_pg_embedding'
                )
            """)
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                cursor.execute("""
                    DELETE FROM langchain_pg_embedding
                    WHERE cmetadata->>'source' = %s;
                """, (doc_name,))
                logging.info(f"Deleted document {doc_name} from langchain_pg_embedding table")
        except Exception as e:
            logging.warning(f"Error deleting from langchain_pg_embedding: {str(e)}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"Document {doc_name} successfully deleted. {chunk_count} chunks removed.")
        return chunk_count
    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        if conn:
            try:
                conn.rollback()
                conn.close()
            except:
                pass
        return 0

def clear_all_documents():
    """
    Clear all documents from the database
    
    Returns:
        int: Number of chunks deleted
    """
    conn = None
    try:
        logging.info(f"Clearing all documents from database")
        
        # Get database connection
        conn = get_db_connection()
        if not conn:
            logging.error("Failed to connect to database for clearing documents")
            return 0
            
        cursor = conn.cursor()
        
        # Get document count before deleting
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        
        if count == 0:
            logging.info("No documents found to delete")
            cursor.close()
            conn.close()
            return 0
            
        # Delete all documents
        cursor.execute("DELETE FROM documents")
        
        # Also clear langchain_pg_embedding if it exists
        try:
            # Check if langchain_pg_embedding table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'langchain_pg_embedding'
                )
            """)
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                cursor.execute("DELETE FROM langchain_pg_embedding")
                logging.info("Cleared all entries from langchain_pg_embedding table")
        except Exception as e:
            logging.warning(f"Error clearing langchain_pg_embedding: {str(e)}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"Cleared all documents from database: {count} chunks deleted")
        return count
    except Exception as e:
        logging.error(f"Error clearing documents: {str(e)}")
        if conn:
            try:
                conn.rollback()
                conn.close()
            except:
                pass
        return 0 

def sanitize_table_name(table_name):
    """
    Sanitize a table name for use in SQL queries, handling case sensitivity correctly.
    
    Args:
        table_name (str): The table name to sanitize
        
    Returns:
        str: The sanitized table name
    """
    # If the table name contains uppercase letters, it needs to be quoted
    if any(c.isupper() for c in table_name):
        # Remove any existing quotes and apply proper quoting
        table_name = table_name.replace('"', '')
        return f'"{table_name}"'
    
    return table_name

def get_connection_pool_stats():
    """
    Get statistics about the database connection pool.
    This is a compatibility function for older code.
    
    Returns:
        dict: A dictionary containing connection pool statistics
    """
    logging.warning("get_connection_pool_stats() called but is now deprecated")
    return {
        "connections": {
            "active": 0,
            "idle": 0,
            "total": 0
        },
        "status": "Connections are now managed directly without a pool"
    } 