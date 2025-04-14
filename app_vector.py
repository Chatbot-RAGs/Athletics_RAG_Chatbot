"""
app_vector.py

This module manages vector operations for document retrieval.
Provides:
- Vector store creation functionality
- Conversion of document content to vector embeddings
- Integration with the database for storing vectorized documents
- Metadata preparation for document retrieval
"""

import streamlit as st
import logging
from app_database import initialize_pgvector, save_document_to_db, get_connection_string
from app_embeddings import embeddings

# Import PGVector for custom extension
from langchain_community.vectorstores.pgvector import PGVector
from sqlalchemy import text, or_, and_
import json
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomPGVector(PGVector):
    """
    A custom PGVector implementation that uses our custom_metadata_match function
    instead of the jsonb_path_match function.
    """
    
    def _and(self, clauses):
        """
        Add the missing _and method required by the filter implementation.
        """
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return and_(*clauses)
            
    def _or(self, clauses):
        """
        Add the missing _or method required by the filter implementation.
        """
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return or_(*clauses)
    
    def _handle_metadata_filter(self, field, operator, value):
        """
        Override the metadata filtering to use our custom_metadata_match function.
        This replaces the jsonb_path_match function that's causing errors.
        """
        if operator == "$eq":
            # Fix ambiguous column reference by specifying the table
            return text("custom_metadata_match(langchain_pg_embedding.cmetadata, :field, :value)").bindparams(
                field=field, value=str(value)
            )
        # Handle other operators as needed
        elif operator == "$in":
            # For $in operator, check if value is in a list
            clauses = []
            for item in value:
                clauses.append(
                    text("custom_metadata_match(langchain_pg_embedding.cmetadata, :field, :value)").bindparams(
                        field=field, value=str(item)
                    )
                )
            return self._or(clauses)
        
        # Fall back to original implementation for other operators
        return super()._handle_metadata_filter(field, operator, value)
    
    def _create_filter_clause(self, filters):
        """
        Create a filter clause based on a dictionary of filters.
        Handles both metadata.field: {"$operator": value} and metadata.field: value formats.
        """
        if not filters:
            return None
            
        # Special handling for our fixed format filter
        if "metadata" in filters and isinstance(filters["metadata"], dict):
            metadata_filters = filters["metadata"]
            clauses = []
            
            for field, operators in metadata_filters.items():
                if isinstance(operators, dict):
                    for operator, value in operators.items():
                        clauses.append(self._handle_metadata_filter(field, operator, value))
                else:
                    # Direct value match (implicit $eq)
                    clauses.append(self._handle_metadata_filter(field, "$eq", operators))
                    
            return self._and(clauses) if clauses else None
        
        # Handle old-style filters
        return super()._create_filter_clause(filters)

def create_vector_store(documents, doc_name):
    """
    Creates a vector store using LangChain Documents.
    Saves documents both to documents table and to langchain_pg_embedding
    for proper vector search.
    
    Args:
        documents: List of Document objects
        doc_name: Name of the document
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize database if needed
        if not initialize_pgvector():
            logger.error("Failed to initialize PGVector database")
            return None

        logger.info(f"Creating vector store for {doc_name} with {len(documents)} chunks")
        
        # Log a sample document for debugging
        if documents and len(documents) > 0:
            sample_doc = documents[0]
            logger.info(f"Sample document content: {sample_doc.page_content[:100]}...")
            logger.info(f"Sample document metadata: {sample_doc.metadata}")
        
        # Save to our documents table for regular SQL operations
        # Extract content and metadata
        chunks = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]

        # Generate embeddings for chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        try:
            embeddings_list = embeddings.embed_documents(chunks)
            logger.info(f"Generated {len(embeddings_list)} embeddings")
            if embeddings_list and len(embeddings_list) > 0:
                logger.info(f"Sample embedding dimensions: {len(embeddings_list[0])}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return None

        # Save to documents table
        if not save_document_to_db(doc_name, chunks, embeddings_list, metadata_list):
            logger.error(f"Failed to save documents to documents table")
            return None
            
        # Now also save to langchain_pg_embedding table using PGVector directly
        try:
            logger.info(f"Saving embeddings to langchain_pg_embedding table...")
            connection_string = get_connection_string()
            
            if not connection_string:
                logger.error("Could not get database connection string")
                return None
                
            # Use our custom PGVector implementation
            vector_store = CustomPGVector.from_documents(
                documents=documents,
                embedding=embeddings,
                connection_string=connection_string,
                collection_name="documents",  # Corresponds to table prefix
                use_jsonb=True
            )
            
            logger.info(f"Successfully created vector store and saved embeddings for {doc_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving to vector store: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error creating vector store for {doc_name}: {str(e)}")
        return None

def get_vector_store():
    """
    Gets the vector store for document retrieval.
    
    Returns:
        CustomPGVector: The custom vector store for document retrieval
    """
    try:
        # Check if database is initialized
        if not initialize_pgvector():
            logger.error("Failed to initialize PGVector database")
            return None
        
        # Get database connection
        connection_string = get_connection_string()
        if not connection_string:
            logger.error("Could not get database connection string")
            return None
        
        # Initialize the vector store using our custom class
        vector_store = CustomPGVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name="documents",
            use_jsonb=True,
            pre_delete_collection=False,  # Don't delete existing collection
            collection_metadata={"extend_existing": True}  # Allow redefining the table
        )
        
        return vector_store
    except Exception as e:
        logger.error(f"Error getting vector store: {str(e)}")
        return None
