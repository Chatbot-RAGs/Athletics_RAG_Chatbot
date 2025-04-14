# Athletics RAG Chatbot Functions Reference

This document provides a comprehensive list of functions available in each module of the RAG system, making it easier to fix import issues and understand the architecture.

## app.py
Main application entry point with tab organization and global settings.
- `connect_postgres()`: Returns a database connection using get_db_connection()
- `main()`: Creates tabs and initializes the application
- `cleanup()`: Performs cleanup operations when the app is shutting down

## app_rag.py
Orchestrates the RAG functionality.
- `process_user_query(query, doc_name, limit)`: Processes a user query using the RAG pipeline

## app_retrieval.py
Implements advanced RAG techniques with parent-child document structure.
- `implement_parent_document_retriever(documents, doc_name)`: Sets up the parent-child document structure
- `fetch_parent_context(child_docs, parent_limit)`: Retrieves parent documents for context enrichment
- `hybrid_retriever(query, vector_store, doc_name, limit)`: Combines vector search and semantic SQL search

## app_documents.py
Processes and manages document ingestion and chunking.
- `get_pdf_pages(pdf_docs, extract_tables)`: Extracts pages from PDF files
- `process_pdfs_in_parallel(pdf_docs, max_workers)`: Processes multiple PDF documents in parallel
- `get_text_chunks(pages_data, chunk_size, chunk_overlap)`: Splits text into chunks
- `process_pdf_file(file, save_to_dropbox)`: Processes a PDF file and stores it in the database
- `process_dropbox_document(file_path)`: Processes a document from Dropbox

## app_database.py
Handles database connections and operations.
- `get_db_connection()`: Gets a connection to the PostgreSQL database
- `get_connection_string()`: Gets the connection string for the PostgreSQL database
- `connect_mysql()`: Gets a connection to the MySQL database
- `execute_postgres_query(query, params)`: Executes a query on the PostgreSQL database
- `execute_mysql_query(query, params)`: Executes a query on the MySQL database
- `initialize_database()`: Initializes the database schema
- `initialize_pgvector()`: Initializes the pgvector extension
- `save_document_to_db(doc_name, chunks, embeddings, metadatas)`: Saves document chunks to the database
- `delete_document(doc_name)`: Deletes a document from the database
- `delete_document_from_database(doc_name)`: Actually removes the document from the database
- `make_json_safe(data)`: Makes data safe for JSON serialization
- `sanitize_table_name(name)`: Sanitizes table names
- `check_database_status()`: Checks if the database is available

## app_vector.py
Manages vector embeddings and vector store operations.
- `create_vector_store(documents, doc_name)`: Creates a vector store using LangChain Documents
- `get_vector_store()`: Gets the vector store instance
- `CustomPGVector` class: Extended PGVector class with additional functionality

## app_embeddings.py
Creates and manages embeddings for document chunks.
- `embeddings`: The embedding function used for vector search

## app_search.py
Provides SQL-based search capabilities.
- `natural_language_sql_search(query, doc_name, limit)`: Uses TF-IDF to find semantically similar content
- `simple_sql_search(query, doc_name, limit)`: Fallback search using keywords and fuzzy matching
- `sql_keyword_search(query, doc_name, include_tables, table_boost, limit)`: Direct keyword-based search
- `diagnose_vector_search_issues(doc_name, limit)`: Troubleshooting tool for vector store problems
- `inspect_table_structure()`: Checks the structure of the documents table

## app_ranking.py
Handles document ranking and relevance scoring.
- `rank_docs_by_relevance(docs, query)`: Ranks documents by keyword relevance
- `is_table_oriented_query(query)`: Detects if a query is likely asking for tabular data

## app_multi_search.py
Enables searching across multiple documents.
- `multi_document_search(query, vector_store, doc_names, limit)`: Searches across multiple documents

## app_llm.py
Interfaces with language models.
- `get_answer(question, context, retrieved_docs)`: Gets a response from the LLM based on the query and context
- `get_llm_response(query, context, retrieved_docs)`: Wrapper function for get_answer

## app_functions.py
Provides utility functions.
- `format_retrieval_context(docs)`: Formats retrieved documents into a context for the LLM
- `create_prompt_with_context(query, context)`: Creates a prompt with the query and context
- `extract_and_cite_answer(response, context_docs)`: Extracts an answer from the LLM response and adds citations
- `format_metrics_for_streamlit(metrics)`: Formats metrics for display in Streamlit
- `save_dataframe(df, query, query_type)`: Saves a dataframe to a parquet file
- `get_temp_files()`: Gets list of temporary data files
- `clean_temp_files(keep_current)`: Cleans up temporary data files
- `get_answer_from_documents(query, docs, llm)`: Gets an answer from the LLM based on documents

## app_docs.py
Manages document metadata.
- `get_document_stats()`: Gets statistics about documents in the database
- `get_document_info(doc_name)`: Gets information about a specific document

## app_dropbox.py
Handles Dropbox integration.
- `get_dropbox_client()`: Gets a Dropbox client
- `is_dropbox_configured()`: Checks if Dropbox is configured
- `list_dropbox_folders(path)`: Lists folders in Dropbox
- `list_dropbox_pdf_files(folder)`: Lists PDF files in a Dropbox folder
- `download_dropbox_file(file_path)`: Downloads a file from Dropbox
- `save_file_to_dropbox(file, path)`: Saves a file to Dropbox
- `create_file_like_object(file_content, file_name)`: Creates a file-like object
- `set_dropbox_on_demand(status)`: Sets Dropbox on-demand status
- `get_dropbox_usage_stats()`: Gets Dropbox usage statistics
- `initialize_dropbox_client()`: Initializes Dropbox client

## openrouter_utils.py
Provides utilities for interacting with OpenRouter.
- `nl_to_sql(query, schema_info, db_type)`: Converts natural language query to SQL
- `github_analyze(repo_name, task)`: Uses OpenRouter API to analyze GitHub repository data 