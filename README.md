# Aspire Academy Athletics Dashboard

A comprehensive dashboard for analyzing sports databases, with advanced AI-powered chat and document search capabilities.

## Overview

This application integrates database querying, document analysis, and AI-powered features into a unified dashboard. It's built with Streamlit and uses DeepSeek Chat v3 model through OpenRouter for AI capabilities.

## Documentation

For detailed technical documentation about the RAG (Retrieval Augmented Generation) system, please refer to [RAG_DOCUMENTATION.md](RAG_DOCUMENTATION.md). This document provides comprehensive information about:

- System architecture and components
- RAG implementation details
- Document processing and vector embedding
- Hybrid search approach
- Query processing flow
- Technical deep dive into RAG components

## Core Files

- **app.py**: Main application entry point with tab organization and global settings
- **app_rag.py**: Advanced RAG implementation with hybrid retrieval techniques
- **app_vector.py**: Vector store management for document embeddings
- **app_documents.py**: Document processing and management functions
- **app_database.py**: Database connection and query execution utilities
- **app_functions.py**: Shared utility functions
- **app_dropbox.py**: Dropbox integration for document storage
- **app_embeddings.py**: Embedding generation for vector search
- **app_retrieval.py**: Implements advanced RAG techniques
- **app_search.py**: Provides SQL-based search functionality
- **app_ranking.py**: Handles document ranking
- **app_multi_search.py**: Enables searching across documents
- **app_llm.py**: Interfaces with language models
- **app_docs.py**: Manages document metadata

## Tab Functionalities

### 1. RAG Explorer (tab3_rag.py)

The centerpiece of the application - a powerful document search and question-answering interface with:

- **Document Upload**: PDF upload and automatic processing into searchable chunks
- **Document Search**: Two search modes:
  - **Standard Search**: Retrieves relevant document chunks using hybrid retrieval
  - **Multi-Document Search**: Search across multiple complete documents
- **Advanced Question Answering**: Ask questions about retrieved documents with AI-generated answers and source citations
- **Dropbox Integration**: Browse and process documents directly from Dropbox
- **Document Management**: Organize, view, and delete processed documents

### 2. Data Analysis (tab4_analysis.py)

AI-powered analysis of query results with:

- **Current Data Analysis**: Analyze query results with natural language questions
- **Multi-DataFrame Analysis**: Analyze relationships between multiple data sources
- **Backend Options**: Choose between OpenRouter, PandasAI, or hybrid analysis approaches
- **Visualization Generation**: Automatically create visualizations based on queries
- **Follow-up Questions**: Ask additional questions about the analysis results

### 3. Knowledge Bank (tab5_knowledge.py)

Store and retrieve knowledge entries:

- **Knowledge Entry Management**: Create, categorize, and search knowledge entries
- **Import/Export**: Share knowledge between systems
- **Category Management**: Organize entries by customizable categories

### 4. PostgreSQL Explorer (tab1_postgres.py)

Database query interface with:

- **SQL Execution**: Run and save query results
- **Table Browser**: Explore database schema
- **Query History**: Track and reuse previously executed queries

### 5. MySQL Explorer (tab2_mysql.py)

Similar to PostgreSQL explorer but for MySQL databases.

## Advanced RAG Features

The application uses a sophisticated RAG (Retrieval Augmented Generation) system with:

- **Hybrid Retrieval**: Combines vector similarity, SQL keyword search, and table-focused retrieval
- **Parent-Child Document Structure**: Maintains context while enabling precise retrieval
- **Document Chunking**: Recursive character text splitting with optimal chunk sizes
- **Metadata-Rich Storage**: Enhanced document storage with comprehensive metadata
- **Relevance Ranking**: Multiple ranking strategies for retrieved documents
- **Context Formulation**: Intelligent prompt engineering with document context
- **Source Citation**: Answers include citations to source documents and pages
- **Performance Metrics**: Track retrieval statistics for system optimization

## Chat Capabilities

AI-powered chat features include:

- **Document QA**: Ask questions about documents with context-aware responses
- **Data Analysis Chat**: Natural language analysis of database query results
- **Multi-Source Integration**: Combine knowledge from documents and data
- **OpenRouter Integration**: Leverages DeepSeek Chat v3 for high-quality responses
- **Follow-up Questions**: Context-aware follow-up based on previous analyses
- **Visualization Integration**: AI can generate and explain data visualizations

## Environment Setup

The application requires several environment variables for configuration:

- Database credentials (PostgreSQL and MySQL)
- OpenRouter API key for AI capabilities
- Dropbox API credentials (optional)

## Getting Started

1. Clone the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Set up environment variables or a `.env` file
4. Run the application with `streamlit run app.py`

## Dependencies

- Streamlit for UI
- LangChain for RAG capabilities
- PandasAI for data analysis
- OpenRouter for LLM access
- PostgreSQL with pgvector for vector search
- MySQL for relational data
- Dropbox API for document storage

## Debugging

The application logs detailed debug information to `rag_debug.log`. This log file contains information about:

- Document processing
- Vector embedding creation
- Search operations
- Query processing
- LLM interactions
- Performance metrics

Use this log file to troubleshoot issues and optimize system performance.

 