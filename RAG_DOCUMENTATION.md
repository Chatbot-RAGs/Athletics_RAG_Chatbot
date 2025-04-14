# Athletics RAG Chatbot Documentation

## Overview

The Athletics RAG Chatbot is a comprehensive application that combines database querying, document analysis, and AI-powered features into a unified dashboard. It's built with Streamlit and uses advanced RAG (Retrieval Augmented Generation) techniques to provide accurate answers to user queries by retrieving relevant information from both structured databases and unstructured documents.

## System Architecture

### High-Level Architecture

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

### Module Organization

1. **Entry Point**:
   - `app.py`: Main application entry point and tab organization

2. **Tab Modules**:
   - `tabs/tab1_postgres.py`: PostgreSQL Explorer interface
   - `tabs/tab2_mysql.py`: MySQL Explorer interface
   - `tabs/tab3_rag.py`: RAG Explorer interface
   - `tabs/tab4_analysis.py`: Data Analysis interface
   - `tabs/tab5_knowledge.py`: Knowledge Bank interface

3. **Core Modules**:
   - `app_rag.py`: Orchestrates the RAG functionality
   - `app_retrieval.py`: Implements advanced RAG techniques
   - `app_search.py`: Provides SQL-based search functionality
   - `app_vector.py`: Manages vector embeddings and search
   - `app_database.py`: Handles database connections
   - `app_documents.py`: Processes and manages documents
   - `app_embeddings.py`: Creates and manages embeddings
   - `app_llm.py`: Interfaces with language models
   - `app_multi_search.py`: Enables searching across documents
   - `app_ranking.py`: Handles document ranking
   - `app_functions.py`: Provides utility functions
   - `app_dropbox.py`: Handles Dropbox integration
   - `app_docs.py`: Manages document metadata

## RAG System Detailed Documentation

### RAG Overview

The RAG (Retrieval Augmented Generation) system is the core of the application's document search and question-answering capabilities. It combines vector-based semantic search with SQL-based keyword search to provide comprehensive and accurate results.

### RAG Data Flow

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

### RAG Query Processing Flow

1. Query reception by `app_rag.py`
2. Vector store initialization
3. Hybrid retrieval process
4. Result combination and deduplication
5. Relevance ranking
6. Parent context fetching
7. Context formatting
8. LLM response generation
9. Response delivery

### Key RAG Components

#### 1. Document Processing (app_documents.py)
- PDF processing with PyPDF2 and tabula-py
- Table extraction and preservation
- Document chunking with RecursiveCharacterTextSplitter
- Metadata extraction and preservation
- Database storage

#### 2. Vector Embedding (app_embeddings.py, app_vector.py)
- Sentence transformers for embedding creation
- PostgreSQL pgvector storage
- CustomPGVector class with enhanced functionality
- Multiple vector search strategies

#### 3. Semantic SQL Search (app_search.py)
- Natural language SQL search with TF-IDF
- Simple SQL search with keyword matching
- SQL keyword search with table boosting
- Error diagnosis and troubleshooting

#### 4. Hybrid Retrieval (app_retrieval.py)
- Vector search for semantic similarity
- Semantic SQL search for keyword matching
- Table-oriented search for structured data
- Result combination and deduplication
- Metrics tracking

#### 5. Document Ranking (app_ranking.py)
- Multi-factor scoring algorithm
- Exact match detection
- Word boundary consideration
- Position-based scoring
- Table-oriented query detection

#### 6. Parent Context Fetching (app_retrieval.py)
- Parent document retrieval
- Proximity-based selection
- Metadata preservation
- Context enhancement

#### 7. LLM Response Generation (app_llm.py)
- OpenRouter integration with DeepSeek Chat v3
- Context-aware prompting
- Citation formatting
- Follow-up question handling

## Technical Deep Dive: RAG System Components

### 1. Document Chunking Process

#### Dynamic Chunk Size Adjustment
- Short texts (<1000 chars): 300 chars chunks, 30 chars overlap
- Long texts (>5000 chars): 1000 chars chunks, 100 chars overlap
- Medium texts: Default values (500/50)

#### Intelligent Splitting
- Uses RecursiveCharacterTextSplitter
- Hierarchy of separators: paragraphs → lines → sentences → punctuation → words
- Preserves semantic meaning

#### Parent-Child Structure
- Parent chunks (2000 chars): Context preservation
- Child chunks (400 chars): Precise retrieval
- Relationship maintenance

### 2. Vector Embedding Process

#### Model Selection
- `all-MiniLM-L6-v2` model from HuggingFace
- 384-dimensional embeddings
- Balanced performance and accuracy

#### Embedding Creation
- Text to vector conversion
- Semantic meaning capture
- Similarity representation

#### Storage
- PostgreSQL with pgvector
- CustomPGVector class
- Relationship maintenance

### 3. Vector Search Enhancements

#### Parameter Tuning
- k = 20 (results to return)
- fetch_k = 50 (results to consider)
- Enhanced candidate pool

#### Search Strategies
- MMR (Maximum Marginal Relevance) with λ=0.5
- Pure similarity search
- Complementary approaches

#### Metadata Filtering
- Custom SQL functions
- Document name filtering
- Page number filtering

### 4. Query-to-Vector Process

#### Query Embedding
- Text to vector conversion
- 384-dimensional representation
- Semantic meaning capture

#### Similarity Calculation
- Cosine similarity computation
- Vector angle measurement
- Optimized search

#### MMR Reranking
- Relevance-diversity balance
- Iterative selection
- λ parameter control

### 5. Hybrid Search Approach

#### Vector Search
- Semantic understanding
- Embedding model usage
- Relationship capture

#### Natural Language SQL Search
- TF-IDF vectorization
- Term importance calculation
- Cosine similarity

#### Result Combination
- Approach combination
- Deduplication
- Best of both worlds

#### Table-Oriented Detection
- Query pattern recognition
- Table content boosting (5.0x)
- Specialized handling

### 6. Document Ranking

#### Multi-factor Scoring
- Exact keyword matches (+5)
- Position in text (+5 for first sentence)
- Multiple keyword presence (+3 per match)
- Exact phrase matches (+10)
- Heading/beginnings keywords (+8)

#### Source Type Tracking
- Vector search results
- SQL search results
- Fallback results

## Why RAG is Important

1. **Accuracy**: Reduces hallucinations and improves factual accuracy
2. **Context Awareness**: Maintains broader context for coherent responses
3. **Transparency**: Shows sources and derivation process
4. **Efficiency**: Focuses LLM on relevant information
5. **Scalability**: Handles large document collections effectively

## Key Innovations

1. **Hybrid Retrieval**: Combines vector and semantic SQL search
2. **Parent-Child Structure**: Maintains context while enabling precise retrieval
3. **Table-Aware Processing**: Special handling for tabular data
4. **Multi-Document Support**: Simultaneous search across documents
5. **Relevance Ranking**: Sophisticated ranking algorithms
6. **Context Enrichment**: Parent document context provision
7. **Comprehensive Metrics**: Detailed retrieval process tracking

## Real-World Benefits

1. **Time Savings**: Quick information retrieval
2. **Improved Decision Making**: Accurate, contextually appropriate answers
3. **Knowledge Discovery**: Connection identification
4. **Reduced Cognitive Load**: Information synthesis handling
5. **Institutional Knowledge Preservation**: Easy access and querying
