"""
app_documentation.py

Comprehensive documentation of the Athletics RAG Chatbot application.
This file provides detailed technical documentation of the entire system,
covering both structured data querying (databases) and unstructured data retrieval (documents).

The application provides a unified interface for querying:
1. Structured data in PostgreSQL and MySQL databases using natural language
2. Unstructured data in PDF documents using advanced RAG techniques
3. Combined analysis of both data types for comprehensive insights
"""

#############################################################################
# INTRODUCTION
#############################################################################
"""
# Athletics RAG Chatbot

## Overview

The Athletics RAG Chatbot is a comprehensive application that combines database querying, 
document analysis, and AI-powered features into a unified dashboard. It's built with 
Streamlit and uses advanced RAG (Retrieval Augmented Generation) techniques to provide 
accurate answers to user queries by retrieving relevant information from both structured 
databases and unstructured documents.

The system bridges the gap between traditional database querying and modern document 
retrieval, allowing users to extract insights from both structured and unstructured data 
sources through a single, intuitive interface.

The application consists of five main tabs:
1. PostgreSQL Explorer: Query athletics data stored in PostgreSQL databases using natural language
2. MySQL Explorer: Query athletics data stored in MySQL databases using natural language
3. RAG Explorer: Upload, manage, and query PDF documents using advanced RAG techniques
4. Data Analysis: Analyze and visualize query results
5. Knowledge Bank: Save and retrieve frequently used queries and insights

## Key Features

- Natural Language Database Queries: Convert natural language questions into SQL
- Advanced RAG Implementation: Hybrid retrieval combining vector search and semantic SQL search
- Document Management: Upload, process, and manage PDF documents with automatic table extraction
- Multi-Document Search: Search across multiple documents simultaneously
- Dropbox Integration: Seamlessly access and process documents stored in Dropbox
- Data Visualization: Generate charts and graphs from query results
- Knowledge Management: Save and organize insights for future reference
"""

#############################################################################
# SYSTEM ARCHITECTURE
#############################################################################
"""
# System Architecture

The application follows a modular architecture with clear separation of concerns:

## High-Level Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  User Interface │     │  Core Services  │     │  Data Sources   │
│  (Streamlit)    │────▶│  (Python)       │────▶│  (DB/Files)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Tab Modules    │     │  App Modules    │     │  External APIs  │
│  (UI Logic)     │────▶│  (Core Logic)   │────▶│  (LLM/Dropbox)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Module Organization

1. Entry Point:
   - app.py: Main application entry point and tab organization

2. Tab Modules:
   - tabs/tab1_postgres.py: PostgreSQL Explorer interface
   - tabs/tab2_mysql.py: MySQL Explorer interface
   - tabs/tab3_rag.py: RAG Explorer interface
   - tabs/tab4_analysis.py: Data Analysis interface
   - tabs/tab5_knowledge.py: Knowledge Bank interface

3. Core Modules:
   - app_rag.py: Orchestrates the RAG functionality
   - app_retrieval.py: Implements advanced RAG techniques
   - app_search.py: Provides SQL-based search functionality
   - app_vector.py: Manages vector embeddings and search
   - app_database.py: Handles database connections
   - app_documents.py: Processes and manages documents
   - app_embeddings.py: Creates and manages embeddings
   - app_llm.py: Interfaces with language models
   - app_multi_search.py: Enables searching across documents
   - app_ranking.py: Handles document ranking
   - app_functions.py: Provides utility functions
   - app_dropbox.py: Handles Dropbox integration
   - app_docs.py: Manages document metadata
"""

#############################################################################
# RAG SYSTEM DETAILED DOCUMENTATION
#############################################################################
"""
# RAG System Detailed Documentation

## RAG Overview

The RAG (Retrieval Augmented Generation) system is the core of the application's document 
search and question-answering capabilities. It combines vector-based semantic search with 
SQL-based keyword search to provide comprehensive and accurate results.

The RAG system follows these key principles:

1. **Hybrid Retrieval**: Combines multiple search strategies for better recall
2. **Parent-Child Document Structure**: Maintains context while enabling precise retrieval
3. **Semantic Understanding**: Uses both vector embeddings and TF-IDF for semantic matching
4. **Table-Aware Processing**: Special handling for tabular data
5. **Multi-Document Support**: Search across multiple documents simultaneously
6. **Relevance Ranking**: Sophisticated ranking algorithms for optimal result ordering
7. **Context Enrichment**: Fetches parent documents to provide more context to the LLM

## RAG Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Document       │     │  Document       │     │  Vector         │
│  Upload         │────▶│  Processing     │────▶│  Embedding      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Response       │     │  LLM            │     │  Database       │
│  Generation     │◀────│  Processing     │◀────│  Storage        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
       ▲                                               │
       │                                               │
       │           ┌─────────────────┐                 │
       │           │  User           │                 │
       └───────────│  Query          │─────────────────┘
                   └─────────────────┘
                          │
                          ▼
                   ┌─────────────────┐
                   │  app_rag.py     │
                   │  (Orchestrator) │
                   └─────────────────┘
                          │
                          ▼
       ┌───────────────────────────────────────┐
       │                                       │
       ▼                                       ▼
┌─────────────────┐               ┌─────────────────┐
│  Vector Search  │               │  Semantic SQL   │
│  (app_vector.py)│               │  (app_search.py)│
└─────────────────┘               └─────────────────┘
       │                                       │
       └───────────────────────────────────────┘
                          │
                          ▼
                   ┌─────────────────┐
                   │  Document       │
                   │  Ranking        │
                   └─────────────────┘
                          │
                          ▼
                   ┌─────────────────┐
                   │  Parent Context │
                   │  Fetching       │
                   └─────────────────┘
                          │
                          ▼
                   ┌─────────────────┐
                   │  LLM Response   │
                   │  Generation     │
                   └─────────────────┘
```

## RAG Query Processing Flow

When a user submits a query, the following steps occur:

1. The query is received by `app_rag.py` through the `process_user_query` function
2. `app_rag.py` initializes the vector store using `app_vector.py`
3. The query is passed to `hybrid_retriever` in `app_retrieval.py`
4. `hybrid_retriever` performs two parallel searches:
   - Vector search using the vector store
   - Semantic SQL search using `natural_language_sql_search` in `app_search.py`
5. Results from both searches are combined and deduplicated
6. The combined results are ranked by relevance using `rank_docs_by_relevance` in `app_ranking.py`
7. Parent context is fetched for each retrieved document using `fetch_parent_context`
8. The context is formatted for the LLM using `format_retrieval_context`
9. The formatted context and query are sent to the LLM using `get_llm_response`
10. The LLM generates a response based on the context and query
11. The response, context, and metrics are returned to the user

## Key RAG Components

### 1. Document Processing (app_documents.py)

The document processing pipeline handles the ingestion, parsing, and chunking of documents:

- PDF documents are processed using PyPDF2 and tabula-py
- Tables are automatically extracted and preserved
- Documents are split into chunks using RecursiveCharacterTextSplitter
- Metadata is extracted and preserved
- Chunks are stored in the database

### 2. Vector Embedding (app_embeddings.py, app_vector.py)

The vector embedding system creates and manages embeddings for document chunks:

- Sentence transformers are used to create embeddings
- Embeddings are stored in PostgreSQL using pgvector
- CustomPGVector class extends PGVector with additional functionality
- Vector search uses multiple strategies (MMR, similarity)

### 3. Semantic SQL Search (app_search.py)

The semantic SQL search system provides a fallback when vector search fails:

- `natural_language_sql_search`: Uses TF-IDF vectorization for semantic matching
- `simple_sql_search`: Uses keyword matching and trigram fuzzy matching
- `sql_keyword_search`: Direct keyword-based search with table boosting

### 4. Hybrid Retrieval (app_retrieval.py)

The hybrid retrieval system combines multiple search strategies:

- Vector search for semantic similarity
- Semantic SQL search for keyword matching
- Table-oriented search for structured data
- Results are combined and deduplicated
- Metrics are tracked for each search strategy

### 5. Document Ranking (app_ranking.py)

The document ranking system determines the most relevant documents:

- Multi-factor scoring algorithm
- Considers exact matches, word boundaries, position in text
- Detects table-oriented queries for specialized handling
- Sorts documents by relevance score

### 6. Parent Context Fetching (app_retrieval.py)

The parent context fetching system provides more context to the LLM:

- Fetches parent documents for each retrieved chunk
- Uses proximity-based selection (previous/next pages)
- Preserves metadata and relationships
- Enhances context understanding for the LLM

### 7. LLM Response Generation (app_llm.py)

The LLM response generation system creates answers based on the context:

- Uses OpenRouter to access DeepSeek Chat v3
- Creates prompts with context and query
- Formats responses with citations
- Handles follow-up questions with context preservation
"""

#############################################################################
# DETAILED MODULE DOCUMENTATION
#############################################################################
"""
# Detailed Module Documentation

## app.py

The main entry point for the application. It initializes the Streamlit interface,
sets up the tabs, and handles global state and settings.

Key functions:
- `main()`: Creates the tabs and initializes the application
- `cleanup()`: Performs cleanup operations when the app is shutting down

## app_rag.py

The central orchestrator for the RAG functionality. It coordinates between document
processing, retrieval, and LLM components.

Key functions:
- `process_user_query(query, doc_name, limit)`: Processes a user query using the RAG pipeline

## app_retrieval.py

Implements advanced RAG techniques based on the Parent Document Retriever pattern.
It creates parent documents (large chunks) and child documents (small chunks) to maintain
context while enabling precise retrieval.

Key components:
- `DocumentCollection`: Extended list class for tracking search metrics
- `implement_parent_document_retriever(documents, doc_name)`: Sets up the parent-child document structure
- `fetch_parent_context(child_docs, parent_limit)`: Retrieves parent documents for context enrichment
- `hybrid_retriever(query, vector_store, doc_name, limit)`: Combines vector search and semantic SQL search

## app_search.py

Provides advanced search capabilities for the RAG system, including natural language
SQL search, simple SQL search, and keyword-based search.

Key functions:
- `natural_language_sql_search(query, doc_name, limit)`: Uses TF-IDF to find semantically similar content
- `simple_sql_search(query, doc_name, limit)`: Fallback search using keywords and fuzzy matching
- `sql_keyword_search(query, doc_name, include_tables, table_boost, limit)`: Direct keyword-based search
- `diagnose_vector_search_issues(doc_name, limit)`: Troubleshooting tool for vector store problems

## app_vector.py

Manages vector embeddings and vector store operations. It provides a custom implementation
of PGVector with additional functionality.

Key functions:
- `get_vector_store()`: Gets the vector store instance
- `create_vector_store(connection_string)`: Creates a new vector store
- `CustomPGVector`: Extended PGVector class with additional functionality

## app_ranking.py

Handles document ranking and relevance scoring for the RAG system. It provides algorithms
to determine the most relevant documents for a given query.

Key functions:
- `rank_docs_by_relevance(docs, query)`: Ranks documents by keyword relevance
- `is_table_oriented_query(query)`: Detects if a query is likely asking for tabular data

## app_multi_search.py

Enables searching across multiple documents simultaneously, combining results from both
vector search and semantic SQL search for comprehensive retrieval.

Key functions:
- `multi_document_search(query, vector_store, doc_names, limit)`: Searches across multiple documents

## app_documents.py

Processes and manages document ingestion and chunking. It handles PDF parsing, table
extraction, and document storage.

Key functions:
- `process_pdf_file(file)`: Processes a PDF file and stores it in the database
- `get_pdf_pages(pdf_file)`: Extracts pages from a PDF file
- `get_text_chunks(text, chunk_size, chunk_overlap)`: Splits text into chunks
- `process_dropbox_document(file_path)`: Processes a document from Dropbox

## app_embeddings.py

Creates and manages embeddings for document chunks. It uses sentence transformers to
create embeddings for vector search.

Key components:
- `embeddings`: The embedding function used for vector search

## app_llm.py

Interfaces with language models for generating responses. It uses OpenRouter to access
DeepSeek Chat v3.

Key functions:
- `get_llm_response(query, context)`: Gets a response from the LLM based on the query and context

## app_database.py

Handles database connections and operations. It provides functions for connecting to
PostgreSQL and MySQL databases.

Key functions:
- `get_db_connection()`: Gets a connection to the PostgreSQL database
- `get_connection_string()`: Gets the connection string for the PostgreSQL database
- `connect_mysql()`: Gets a connection to the MySQL database
- `execute_postgres_query(query, params)`: Executes a query on the PostgreSQL database
- `execute_mysql_query(query, params)`: Executes a query on the MySQL database

## app_functions.py

Provides utility functions used across the application. It includes functions for
formatting context, creating prompts, and extracting answers.

Key functions:
- `format_retrieval_context(docs)`: Formats retrieved documents into a context for the LLM
- `create_prompt_with_context(query, context)`: Creates a prompt with the query and context
- `extract_and_cite_answer(response, docs)`: Extracts an answer from the LLM response and adds citations
- `format_metrics_for_streamlit(metrics)`: Formats metrics for display in Streamlit

## app_dropbox.py

Handles Dropbox integration for document storage and retrieval. It provides functions
for accessing and processing documents stored in Dropbox.

Key functions:
- `get_dropbox_client()`: Gets a Dropbox client
- `list_dropbox_folders(path)`: Lists folders in Dropbox
- `list_dropbox_pdf_files(folder)`: Lists PDF files in a Dropbox folder
- `download_dropbox_file(file_path)`: Downloads a file from Dropbox

## app_docs.py

Manages document metadata and provides functions for retrieving document information.

Key functions:
- `get_document_stats()`: Gets statistics about documents in the database
- `get_document_info(doc_name)`: Gets information about a specific document

## tabs/tab3_rag.py

The RAG Explorer interface. It provides a user interface for uploading, managing, and
querying documents using the RAG system.

Key functions:
- `show_rag_tab()`: Displays the RAG Explorer tab content
"""

#############################################################################
# RAG SYSTEM IMPORTANCE AND BENEFITS
#############################################################################
"""
# RAG System Importance and Benefits

## Why RAG is Important

The RAG (Retrieval Augmented Generation) system is a critical component of the application
for several reasons:

1. **Accuracy**: By retrieving relevant documents before generating a response, RAG ensures
   that the LLM has access to accurate and up-to-date information. This reduces hallucinations
   and improves the factual accuracy of responses.

2. **Context Awareness**: The parent-child document structure and context fetching mechanisms
   ensure that the LLM has access to the broader context surrounding the retrieved information.
   This leads to more coherent and contextually appropriate responses.

3. **Transparency**: RAG provides transparency by showing the sources used to generate the
   response. This allows users to verify the information and understand how the answer was derived.

4. **Efficiency**: By focusing the LLM on relevant information, RAG reduces the computational
   resources required to generate a response. This leads to faster response times and lower costs.

5. **Scalability**: The hybrid retrieval approach allows the system to scale to large document
   collections without sacrificing accuracy or performance.

## Key Innovations in Our RAG Implementation

Our RAG implementation includes several innovative features that enhance its performance:

1. **Hybrid Retrieval**: By combining vector search with semantic SQL search, our system
   achieves higher recall than either approach alone. This is particularly important for
   queries that may not have exact keyword matches in the documents.

2. **Parent-Child Document Structure**: Our implementation of the Parent Document Retriever
   pattern allows us to maintain context while enabling precise retrieval. This is crucial
   for understanding complex documents with interdependent sections.

3. **Table-Aware Processing**: Our system includes special handling for tabular data, which
   is often challenging for traditional RAG systems. This includes table extraction, table
   boosting in search, and table-oriented query detection.

4. **Multi-Document Support**: Our system can search across multiple documents simultaneously,
   which is essential for answering questions that span multiple sources.

5. **Relevance Ranking**: Our sophisticated ranking algorithms ensure that the most relevant
   documents are presented to the LLM first. This improves the quality of the generated response.

6. **Context Enrichment**: Our parent context fetching mechanism provides additional context
   to the LLM, which is crucial for understanding complex documents with interdependent sections.

7. **Comprehensive Metrics**: Our system tracks detailed metrics about the retrieval process,
   which helps users understand how the answer was derived and identify potential issues.

## Real-World Benefits

The advanced RAG system provides several real-world benefits for users:

1. **Time Savings**: Users can quickly find information in large document collections without
   having to read through entire documents.

2. **Improved Decision Making**: By providing accurate and contextually appropriate answers,
   the system helps users make better decisions based on the information in their documents.

3. **Knowledge Discovery**: The system can uncover connections between different documents
   that users might not have noticed otherwise.

4. **Reduced Cognitive Load**: By handling the retrieval and synthesis of information, the
   system reduces the cognitive load on users, allowing them to focus on higher-level tasks.

5. **Institutional Knowledge Preservation**: The system helps preserve institutional knowledge
   by making it easily accessible and queryable.
"""

#############################################################################
# DATABASE QUERYING SYSTEM
#############################################################################
"""
# Database Querying System

The application provides natural language querying capabilities for both PostgreSQL and MySQL 
databases through dedicated tabs. This system allows users to interact with structured data 
using plain English questions instead of writing SQL queries.

## Natural Language to SQL Process

The database querying system follows these steps to convert natural language to SQL:

1. **Query Reception**: User submits a natural language question through the interface
2. **LLM Processing**: The query is sent to the DeepSeek Chat v3 model via OpenRouter
3. **SQL Generation**: The LLM generates appropriate SQL based on the database schema
4. **Query Execution**: The generated SQL is executed against the selected database
5. **Result Formatting**: Results are formatted as pandas/polars DataFrames for display
6. **Result Storage**: Query results are saved as parquet files for later analysis

## PostgreSQL Explorer (Tab 1)

The PostgreSQL Explorer tab provides access to athletics performance data:

1. **Schema Understanding**: The system loads table schemas to understand the database structure
2. **Context Building**: Database schema is included in the prompt to the LLM
3. **Query Validation**: Generated SQL is validated before execution to prevent errors
4. **Result Visualization**: Query results can be displayed as tables or visualizations
5. **Query History**: Previous queries are saved for reuse and reference

## MySQL Explorer (Tab 2)

The MySQL Explorer tab follows a similar pattern but is optimized for MySQL syntax:

1. **Connection Management**: Handles MySQL-specific connection parameters
2. **Dialect Adaptation**: Adjusts prompts to generate MySQL-compatible SQL
3. **Performance Optimization**: Includes query optimization hints for MySQL
4. **Schema Exploration**: Provides tools to explore the MySQL database structure
5. **Data Export**: Allows exporting query results in various formats

## Data Flow to Analysis Tab

Query results from both database tabs can be seamlessly passed to the Data Analysis tab (Tab 4):

1. **Data Persistence**: Query results are saved as parquet files in the temp_data directory
2. **Session State**: File references are stored in Streamlit's session state
3. **Cross-Tab Communication**: The Analysis tab can access these saved datasets
4. **Combined Analysis**: Multiple datasets from different sources can be analyzed together
5. **Contextual Awareness**: The analysis LLM receives context about data sources and schemas

## LLM Integration for Database Querying

The database querying system uses specialized LLM prompting techniques:

1. **Schema-Aware Prompting**: Database schema is included in the prompt
2. **Few-Shot Examples**: Example queries and their SQL translations are provided
3. **Constraint Communication**: Database constraints and relationships are explained
4. **Error Handling**: The system can recover from SQL errors with refined prompts
5. **Query Optimization**: The LLM is instructed to generate efficient queries

This approach allows users to interact with complex databases using natural language,
making data exploration accessible to non-technical users while still providing
the power and flexibility of SQL.
"""

#############################################################################
# DATA ANALYSIS SYSTEM
#############################################################################
"""
# Data Analysis System

The Data Analysis tab (Tab 4) provides AI-powered analysis of query results from both
database queries and document searches. It serves as the analytical brain of the application,
helping users derive insights from their data.

## Data Analysis Flow

The analysis process follows these steps:

1. **Data Selection**: User selects datasets from previous database queries or uploads new data
2. **Analysis Question**: User formulates a natural language question about the data
3. **Context Building**: The system prepares context including data schema and sample rows
4. **LLM Processing**: The question and context are sent to the DeepSeek Chat v3 model
5. **Analysis Generation**: The LLM generates insights, explanations, and visualization code
6. **Visualization Rendering**: Generated visualization code is executed and displayed
7. **Follow-up Analysis**: User can ask follow-up questions with the context preserved

## Integration with Database Tabs

The Data Analysis tab seamlessly integrates with the database querying tabs:

1. **Data Passing**: Query results from PostgreSQL and MySQL tabs are automatically available
2. **Metadata Preservation**: Column types, source information, and query context are preserved
3. **Cross-Database Analysis**: Data from different databases can be analyzed together
4. **Historical Access**: Previous query results remain available for analysis
5. **Contextual Awareness**: The analysis system understands the origin and meaning of the data

## Integration with RAG Explorer

The Data Analysis tab can also work with information retrieved from documents:

1. **Structured Extraction**: Table data extracted from documents can be analyzed
2. **Mixed-Source Analysis**: Document data can be analyzed alongside database data
3. **Context-Aware Analysis**: Document metadata informs the analysis process
4. **Citation Tracking**: Analysis results can cite document sources
5. **Insight Enrichment**: Document context enhances numerical analysis

## LLM Prompting for Analysis

The system uses specialized prompting techniques for data analysis:

1. **Data Description**: Detailed schema information and sample data are provided
2. **Analysis Guidance**: The LLM is guided to perform specific types of analysis
3. **Visualization Direction**: Instructions for creating appropriate visualizations
4. **Statistical Rigor**: Prompts encourage statistically sound analysis methods
5. **Explanation Requirements**: The LLM must explain its analytical process and findings

## Visualization Capabilities

The analysis system can generate various types of visualizations:

1. **Statistical Charts**: Histograms, scatter plots, bar charts, line graphs
2. **Comparative Visualizations**: Side-by-side comparisons, difference charts
3. **Time Series Analysis**: Trend lines, seasonal decomposition, forecasting
4. **Relationship Mapping**: Correlation matrices, network graphs
5. **Interactive Elements**: Tooltips, zooming, filtering options

This powerful analysis system transforms raw data into actionable insights,
helping users understand patterns, trends, and relationships in their athletics data.
"""

#############################################################################
# CONCLUSION
#############################################################################
"""
# Conclusion

The Athletics RAG Chatbot is a sophisticated application that combines advanced RAG techniques
with database querying and data analysis capabilities. Its modular architecture and comprehensive
feature set make it a powerful tool for exploring athletics data and documents.

The system bridges the gap between structured and unstructured data:

1. **Database Querying**: Natural language access to PostgreSQL and MySQL databases
2. **Document Retrieval**: Advanced RAG techniques for searching and querying PDF documents
3. **Integrated Analysis**: Combined analysis of both structured and unstructured data
4. **Knowledge Management**: Saving and organizing insights for future reference

The application's five tabs provide a unified interface for exploring different types of data:

1. PostgreSQL Explorer: For querying structured data in PostgreSQL databases
2. MySQL Explorer: For querying structured data in MySQL databases
3. RAG Explorer: For searching and querying unstructured data in documents
4. Data Analysis: For analyzing and visualizing query results
5. Knowledge Bank: For saving and retrieving frequently used queries and insights

Together, these components create a powerful tool for athletics data exploration, analysis,
and document retrieval, enabling users to extract comprehensive insights from all available
data sources through a single, intuitive interface.
"""

#############################################################################
# TECHNICAL DEEP DIVE: RAG SYSTEM COMPONENTS
#############################################################################
"""
# Technical Deep Dive: RAG System Components

## 1. Document Chunking Process

The chunking process is a critical first step in the RAG pipeline:

1. **Dynamic Chunk Size Adjustment**:
   - The system analyzes the average text length of your documents
   - For short texts (<1000 chars): Uses smaller chunks (300 chars) with smaller overlap (30 chars)
   - For long texts (>5000 chars): Uses larger chunks (1000 chars) with larger overlap (100 chars)
   - For medium texts: Uses default values (500/50)

2. **Intelligent Splitting**:
   - Uses `RecursiveCharacterTextSplitter` with a hierarchy of separators:
   - Prioritizes natural boundaries: paragraphs → lines → sentences → punctuation → words
   - This preserves semantic meaning better than arbitrary splitting

3. **Parent-Child Structure**:
   - Creates two types of chunks:
     - Parent chunks (2000 chars): Larger for context preservation
     - Child chunks (400 chars): Smaller for precise retrieval
   - Maintains relationships between them for context enrichment

## 2. Vector Embedding Process

The embedding process converts text to numerical vectors:

1. **Model Selection**:
   - Uses the `all-MiniLM-L6-v2` model from HuggingFace
   - Creates 384-dimensional embeddings that capture semantic meaning
   - This model balances performance and accuracy

2. **Embedding Creation**:
   - Each document chunk is converted to a vector representation
   - These vectors capture the semantic meaning of the text
   - Similar meanings have similar vector representations

3. **Storage**:
   - Embeddings are stored in PostgreSQL using pgvector extension
   - Uses a custom `CustomPGVector` class that enhances metadata filtering
   - Maintains relationships between embeddings and original text

## 3. Vector Search Enhancements

The key improvements we made to vector search:

1. **Aggressive Parameter Tuning**:
   - Increased `k` parameter to 20 (number of results to return)
   - Increased `fetch_k` parameter to 50 (number of results to consider before diversity filtering)
   - This provides more candidates for the diversity algorithm to work with

2. **Multiple Search Strategies**:
   - **MMR (Maximum Marginal Relevance)** with λ=0.5:
     - Balances relevance and diversity
     - λ controls the trade-off (higher = more diversity)
   - **Pure similarity search**:
     - Focuses solely on semantic similarity
     - Complements MMR by finding highly relevant but potentially similar results

3. **Metadata Filtering**:
   - Enhanced metadata filtering with custom SQL functions
   - Allows precise filtering by document name, page number, etc.
   - Implemented in `CustomPGVector` class

## 4. Query-to-Vector Process

When a user submits a query, here's how it's processed:

1. **Query Embedding**:
   - The query text is converted to a vector using the same embedding model
   - This creates a 384-dimensional representation of the query's meaning

2. **Similarity Calculation**:
   - The system calculates cosine similarity between the query vector and all document vectors
   - Cosine similarity measures the angle between vectors (1.0 = identical, 0.0 = unrelated)
   - This is extremely fast due to pgvector's optimized similarity search

3. **MMR Reranking**:
   - After finding similar vectors, MMR reranks them to balance relevance and diversity
   - It iteratively selects documents that are both relevant to the query and different from already selected documents
   - The λ parameter (0.5) controls this balance

## 5. Hybrid Search Approach

The most significant improvement is the hybrid search approach:

1. **Vector Search** (semantic understanding):
   - Finds documents with similar meaning even with different words
   - Uses the embedding model to capture semantic relationships

2. **Natural Language SQL Search** (TF-IDF based):
   - Uses TF-IDF vectorization to find term importance
   - Calculates cosine similarity between TF-IDF vectors
   - Particularly effective for keyword matching and rare terms

3. **Result Combination**:
   - Results from both approaches are combined and deduplicated
   - This gives you the best of both worlds:
     - Vector search for semantic understanding
     - TF-IDF search for keyword precision

4. **Table-Oriented Detection**:
   - Detects if queries are asking for tabular data
   - Applies special boosting to table content (5.0x boost factor)
   - Uses specialized patterns to identify table-related queries

## 6. Document Ranking

After retrieval, documents are ranked by relevance:

1. **Multi-factor Scoring Algorithm**:
   - Exact keyword matches (with word boundaries)
   - Position in text (keywords in first sentence get +5 points)
   - Multiple keyword presence (documents with multiple keywords get +3 points per match)
   - Exact phrase matches (+10 points)
   - Keywords in document headings/beginnings (+8 points)

2. **Source Type Tracking**:
   - Each result is tagged with its source (vector, SQL, or fallback)
   - This helps understand which search method found each result

This comprehensive approach ensures you get both semantically relevant results (from vector search) and keyword-precise results (from TF-IDF search), giving you the best possible answers to your queries.
"""
