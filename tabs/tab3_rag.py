"""
RAG Explorer tab for Aspire Academy Athletics Dashboard.
Provides document search and question answering using the RAG (Retrieval Augmented Generation) approach.
"""

import os
import logging
import streamlit as st
import traceback
import json
import time
import uuid
import re
from datetime import datetime
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
import tiktoken

# Import RAG-related modules
from app_database import initialize_pgvector, load_documents_from_database, delete_document, clear_all_documents, get_db_connection
from app_documents import process_pdf_file, get_pdf_pages, get_text_chunks, process_dropbox_document
from app_vector import get_vector_store, create_vector_store
from app_retrieval import hybrid_retriever, DocumentCollection
from app_multi_search import multi_document_search
from app_functions import format_retrieval_context
from app_dropbox import (
    is_dropbox_configured, get_dropbox_client, download_dropbox_file, 
    create_file_like_object, create_dropbox_folder, list_dropbox_folders, 
    delete_dropbox_file, list_dropbox_pdf_files
)
from app_functions import save_dataframe
from app_docs import get_document_stats, get_document_info
from app_llm import get_llm_response

# Set up logging
log_file = "rag_debug.log"
file_handler = logging.FileHandler(log_file)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.DEBUG)

# Helper function for consistent RAG debug logging
def rag_debug_log(message):
    """Helper function to log RAG-specific debug messages with a timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    debug_msg = f"{timestamp} - RAG_DEBUG - {message}"
    logging.debug(debug_msg)
    print(debug_msg)  # Also print to console/stdout for immediate feedback

def show_rag_tab():
    """Display the RAG Explorer tab content."""
    
    # Reset session state for document selection to avoid multiselect errors
    # This is a temporary fix to clear out any invalid selections 
    # from previous format (pages vs chunks)
    if st.session_state.get('selected_rag_documents'):
        st.session_state.selected_rag_documents = []
    
    st.header("RAG Explorer")
    
    # Initialize session state variables
    if 'selected_docs_for_deletion' not in st.session_state:
        st.session_state.selected_docs_for_deletion = None
    if 'show_delete_all_warning' not in st.session_state:
        st.session_state.show_delete_all_warning = False
    if 'selected_rag_documents' not in st.session_state:
        st.session_state.selected_rag_documents = []
    if 'last_rag_question' not in st.session_state:
        st.session_state.last_rag_question = ""
    if 'last_rag_answer' not in st.session_state:
        st.session_state.last_rag_answer = ""
    
    # Create three columns for a clearer info panel at the top
    info_col1, info_col2, info_col3 = st.columns([1, 1, 1])
    with info_col1:
        st.info("📚 Select documents →")
    with info_col2:
        st.info("❓ Ask questions about your documents →")
    with info_col3:
        st.info("📝 Get AI-powered answers →")
    
    # Debug mode toggle in sidebar
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False, help="Enable detailed logging for debugging")
    
    if debug_mode:
        st.sidebar.success("Debug mode enabled - logs will be written to rag_debug.log")
    
    try:
        # Initialize vector store and PGVector
        vector_store = get_vector_store()
        if vector_store is None:
            st.error("Failed to initialize vector store. Please check your database configuration.")
            return
        
        initialize_pgvector()
        
        # Create tabs for different RAG operations
        qa_tab, upload_tab, manage_tab, dropbox_tab = st.tabs([
            "Document Q&A", "Upload Documents", "Document Management", "Dropbox Integration"
        ])
        
        # Document Q&A Tab (New primary tab)
        with qa_tab:
            st.subheader("📝 Ask Questions About Your Documents")
            
            # Load available documents
            docs_in_db = load_documents_from_database()
            
            if len(docs_in_db) == 0:
                st.warning("⚠️ No documents found in the database. Please upload documents first using the 'Upload Documents' tab.")
            else:
                # Show document selection section
                st.markdown("### 📚 Select Documents")
                
                # Create two columns for selection type and actual selection
                select_col1, select_col2 = st.columns([1, 3])
                
                with select_col1:
                    selection_mode = st.radio(
                        "Selection Mode:",
                        ["All Documents", "Specific Documents"],
                        index=0,
                        help="Search across all documents or select specific ones"
                    )
                
                selected_doc_ids = []
                if selection_mode == "Specific Documents":
                    with select_col2:
                        # Format document options to show as Document Name (Pages)
                        doc_info = {}
                        for doc_name, doc_data in docs_in_db.items():
                            doc_info[doc_name] = {
                                'id': doc_name,  # Use doc_name as id
                                'count': doc_data.get('chunk_count', 0)  # Use chunk count
                            }
                        
                        doc_options = [f"{name} ({info['count']} chunks)" for name, info in doc_info.items()]
                        
                        # Clear selected documents if format has changed (from pages to chunks)
                        if st.session_state.selected_rag_documents:
                            # Check if any selected document has old format (contains "pages")
                            if any("pages" in doc for doc in st.session_state.selected_rag_documents):
                                st.session_state.selected_rag_documents = []
                        
                        selected_docs = st.multiselect(
                            "Select documents to query:",
                            options=doc_options,
                            default=st.session_state.selected_rag_documents,
                            help="Choose which documents to search for answers"
                        )
                        
                        # Save selection to session state
                        st.session_state.selected_rag_documents = selected_docs
                        
                        # Get the document IDs from the selection
                        for selection in selected_docs:
                            # Extract document name from the format "name (X chunks)"
                            doc_name = selection.split(" (")[0]
                            # Use the document name itself as the ID
                            selected_doc_ids.append(doc_name)
                        
                        st.info(f"Selected document IDs: {selected_doc_ids}")
                
                # Question form with primary and follow-up
                with st.form("document_qa_form"):
                    st.markdown("### ❓ Ask Your Question")
                    
                    question = st.text_area(
                        "What would you like to know about these documents?",
                        value=st.session_state.last_rag_question,
                        placeholder="e.g., What are the rules of Shot Put? or What training methods are recommended for sprinters?",
                        height=100
                    )
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        submit_button = st.form_submit_button("🔍 Get Answer", type="primary", use_container_width=True)
                    with col2:
                        show_sources = st.checkbox("Show source documents in the answer", value=True)
                
                # Process question when form is submitted
                if submit_button and question:
                    with st.spinner("Searching documents and generating answer..."):
                        try:
                            st.session_state.last_rag_question = question
                            
                            # Debug log
                            st.info(f"Searching for: '{question}' in documents mode: {selection_mode}")
                            st.info(f"Selected document IDs: {selected_doc_ids}")
                            
                            # Choose retrieval method based on selection mode
                            if selection_mode == "All Documents":
                                try:
                                    st.write(f"Searching for: '{question}' in documents mode: All Documents")
                                    # Use None for doc_names to search all documents
                                    # Get vector store first
                                    vector_store = get_vector_store()
                                    if not vector_store:
                                        st.error("Failed to initialize vector store")
                                        return
                                    
                                    results = multi_document_search(query=question, vector_store=vector_store, doc_names=None)
                                    
                                except Exception as e:
                                    st.error(f"Error during search: {str(e)}")
                                    logging.error(f"Error during search: {str(e)}", exc_info=True)
                                    return
                            else:
                                # Specific documents mode
                                selected_doc_ids = []
                                for selection in selected_docs:
                                    # Extract document name from the format "name (X chunks)"
                                    doc_name = selection.split(" (")[0]
                                    # Use the document name itself as the ID
                                    selected_doc_ids.append(doc_name)
                                
                                if not selected_doc_ids:
                                    st.warning("Please select at least one document")
                                    return
                                
                                # Store in session state for follow-up questions
                                st.session_state.selected_doc_ids = selected_doc_ids
                                
                                try:
                                    st.write(f"Searching for: '{question}' in documents mode: Specific Documents")
                                    st.write(f"Selected document IDs: {selected_doc_ids}")
                                    st.write(f"Using multi-document search with {len(selected_doc_ids)} document IDs")
                                    
                                    # Get vector store first
                                    vector_store = get_vector_store()
                                    if not vector_store:
                                        st.error("Failed to initialize vector store")
                                        return
                                    
                                    # Log more details about the selected documents
                                    logger.info(f"Selected document IDs for search: {selected_doc_ids}")
                                    
                                    # Make sure document IDs are properly formatted
                                    # Extract just the document name without the chunk count
                                    clean_doc_ids = []
                                    for doc_id in selected_doc_ids:
                                        # Remove any file extension if present
                                        if '.' in doc_id:
                                            base_name = doc_id.rsplit('.', 1)[0]
                                            clean_doc_ids.append(base_name)
                                        else:
                                            clean_doc_ids.append(doc_id)
                                    
                                    logger.info(f"Clean document IDs for search: {clean_doc_ids}")
                                    
                                    # Call multi_document_search with the cleaned document IDs
                                    results = multi_document_search(query=question, vector_store=vector_store, doc_names=clean_doc_ids)
                                    
                                except Exception as e:
                                    st.error(f"Error during search: {str(e)}")
                                    logging.error(f"Error during search: {str(e)}", exc_info=True)
                                    return
                            
                            st.info(f"Search results count: {len(results) if results else 0}")
                            
                            if results and len(results) > 0:
                                st.success(f"✅ Found {len(results)} relevant document sections")
                                
                                # Prepare context from retrieved documents
                                combined_context = ""
                                for doc in results[:20]:
                                    source = doc.metadata.get('source', 'Unknown')
                                    page = doc.metadata.get('page', 'Unknown')
                                    combined_context += f"\n\nSource: {source}, Page: {page}\nContent: {doc.page_content}"
                                
                                # Get answer using the get_llm_response function
                                answer = get_llm_response(query=question, context=combined_context)
                                st.session_state.last_rag_answer = answer
                                
                                # Display answer in a highlighted box
                                st.markdown("### 📝 Answer")
                                st.markdown(f"""
                                <div style="background-color: #f0f7fb; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;">
                                {answer}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display source documents if requested
                                if show_sources:
                                    with st.expander("📚 View Source Documents"):
                                        st.markdown("#### Sources Used for Answer")
                                        source_cols = st.columns(2)
                                        
                                        for i, doc in enumerate(results[:10]):
                                            col_idx = i % 2
                                            with source_cols[col_idx]:
                                                st.markdown(f"""
                                                <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                                <strong>Source {i+1}</strong>: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', 'Unknown')}
                                                <hr>
                                                {doc.page_content[:250]}{"..." if len(doc.page_content) > 250 else ""}
                                                </div>
                                                """, unsafe_allow_html=True)
                                
                                # Display metrics about the search
                                with st.expander("📊 Search Metrics"):
                                    metrics_cols = st.columns(4)
                                    
                                    with metrics_cols[0]:
                                        st.metric("Total Results", len(results))
                                    with metrics_cols[1]:
                                        # Get count of vector results - each result has metadata with 'source_type'
                                        vector_count = sum(1 for r in results if r.metadata.get('source_type') == 'vector')
                                        st.metric("Vector Results", vector_count)
                                    with metrics_cols[2]:
                                        # Get count of SQL results
                                        sql_count = sum(1 for r in results if r.metadata.get('source_type') == 'sql')
                                        st.metric("SQL Results", sql_count)
                                    with metrics_cols[3]:
                                        st.metric("Documents Searched", len(selected_doc_ids) if selected_doc_ids else "All")
                            else:
                                st.warning("No relevant documents found to answer your question. Try rephrasing or selecting different documents.")
                        except Exception as e:
                            st.error(f"Error during search: {str(e)}")
                            if debug_mode:
                                st.error(traceback.format_exc())
                
                # Follow-up question section (only show if we have a previous answer)
                if st.session_state.last_rag_answer:
                    with st.form("followup_question_form"):
                        st.subheader("🔄 Follow-up Question")
                        followup_question = st.text_area(
                            "Ask a follow-up question about these documents:",
                            placeholder="e.g., Can you explain more about that technique? or What are the measurements for the throwing circle?",
                            height=80
                        )
                        followup_button = st.form_submit_button("Get Follow-up Answer", type="primary")
                    
                    if followup_button and followup_question:
                        with st.spinner("Processing follow-up question..."):
                            try:
                                # Combine previous answer with follow-up for context
                                combined_question = f"Previous answer: {st.session_state.last_rag_answer}\n\nFollow-up question: {followup_question}"
                                
                                # Use the same search approach as the original question
                                if selection_mode == "All Documents":
                                    try:
                                        st.write(f"Searching for follow-up: '{followup_question}' in documents mode: All Documents")
                                        # Use None for doc_names to search all documents
                                        vector_store = get_vector_store()
                                        if not vector_store:
                                            st.error("Failed to initialize vector store")
                                            return
                                            
                                        results = multi_document_search(query=combined_question, vector_store=vector_store, doc_names=None)
                                    except Exception as e:
                                        st.error(f"Error during follow-up search: {str(e)}")
                                        logging.error(f"Error during follow-up search: {str(e)}", exc_info=True)
                                        return
                                else:
                                    # Specific documents mode - use saved doc_ids from session state
                                    selected_doc_ids = st.session_state.get('selected_doc_ids', [])
                                    
                                    if not selected_doc_ids:
                                        st.warning("No documents selected for follow-up question")
                                        return
                                        
                                    try:
                                        st.write(f"Searching for follow-up: '{followup_question}' in documents mode: Specific Documents")
                                        st.write(f"Selected document IDs: {selected_doc_ids}")
                                        
                                        vector_store = get_vector_store()
                                        if not vector_store:
                                            st.error("Failed to initialize vector store")
                                            return
                                        
                                        results = multi_document_search(query=combined_question, vector_store=vector_store, doc_names=selected_doc_ids)
                                    except Exception as e:
                                        st.error(f"Error during follow-up search: {str(e)}")
                                        logging.error(f"Error during follow-up search: {str(e)}", exc_info=True)
                                        return
                                
                                if results and len(results) > 0:
                                    # Prepare context from retrieved documents
                                    combined_context = ""
                                    for doc in results[:20]:
                                        source = doc.metadata.get('source', 'Unknown')
                                        page = doc.metadata.get('page', 'Unknown')
                                        combined_context += f"\n\nSource: {source}, Page: {page}\nContent: {doc.page_content}"
                                    
                                    # Include previous answer in context
                                    full_context = f"Previous question: {st.session_state.last_rag_question}\nPrevious answer: {st.session_state.last_rag_answer}\n\n{combined_context}"
                                    
                                    # Get follow-up answer using the get_llm_response function
                                    followup_answer = get_llm_response(query=followup_question, context=full_context)
                                    
                                    # Update session state with new question/answer
                                    st.session_state.last_rag_question = followup_question
                                    st.session_state.last_rag_answer = followup_answer
                                    
                                    # Display follow-up answer in a highlighted box
                                    st.markdown("### 📝 Follow-up Answer")
                                    st.markdown(f"""
                                    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                                    {followup_answer}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Display source documents if requested
                                    if show_sources:
                                        with st.expander("📚 View Source Documents for Follow-up"):
                                            st.markdown("#### Sources Used for Follow-up Answer")
                                            source_cols = st.columns(2)
                                            
                                            for i, doc in enumerate(results[:10]):
                                                col_idx = i % 2
                                                with source_cols[col_idx]:
                                                    st.markdown(f"""
                                                    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                                    <strong>Source {i+1}</strong>: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', 'Unknown')}
                                                    <hr>
                                                    {doc.page_content[:250]}{"..." if len(doc.page_content) > 250 else ""}
                                                    </div>
                                                    """, unsafe_allow_html=True)
                                else:
                                    st.warning("No relevant documents found to answer your follow-up question. Try rephrasing.")
                            except Exception as e:
                                st.error(f"Error during follow-up: {str(e)}")
                                if debug_mode:
                                    st.error(traceback.format_exc())
        
        # Document Upload Tab
        with upload_tab:
            st.subheader("📤 Upload Documents")
            
            upload_col1, upload_col2 = st.columns([2, 1])
            
            with upload_col1:
                st.write("Upload PDF documents to be processed and stored in the vector database.")
                
                # File uploader for PDF documents
                uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
                
                if uploaded_file is not None:
                    # Display document info in a nicer format
                    st.markdown("### 📄 Document Information")
                    
                    doc_info_cols = st.columns(3)
                    with doc_info_cols[0]:
                        st.metric("Filename", uploaded_file.name)
                    with doc_info_cols[1]:
                        st.metric("Size", f"{uploaded_file.size / 1024:.2f} KB")
                    with doc_info_cols[2]:
                        st.metric("Type", uploaded_file.type)
                    
                    # Process document button with a key to ensure it's unique
                    if st.button("📥 Process and Store Document", key="process_doc_button", type="primary"):
                        with st.spinner("Processing document..."):
                            try:
                                # Process the document
                                success = process_pdf_file(uploaded_file)
                                
                                if success:
                                    st.success(f"✅ Document processed successfully! Document ID: {uploaded_file.name}")
                                    st.info("Document has been added to the vector store and is ready for querying.")
                                    
                                    # Clear the file uploader by forcing a rerun
                                    uploaded_file = None
                                    st.rerun()
                                else:
                                    st.error("Failed to process document.")
                            except Exception as e:
                                st.error(f"Error processing document: {str(e)}")
                                if debug_mode:
                                    st.error(traceback.format_exc())
            
            with upload_col2:
                st.markdown("### 📋 Upload Tips")
                st.markdown("""
                - Only PDF files are supported
                - Each page will be processed separately
                - Document metadata will be extracted
                - Text is split into smaller chunks
                - Vector embeddings are created for search
                """)
        
        # Document Management Tab
        with manage_tab:
            st.subheader("📋 Document Management")
            
            try:
                # List documents in database
                docs = load_documents_from_database()
                
                if docs:
                    manage_col1, manage_col2 = st.columns([3, 1])
                    
                    with manage_col1:
                        st.success(f"✅ Found {len(docs)} document chunks in the database")
                        
                        # Group documents by source
                        doc_sources = {}
                        for doc_name, doc_info in docs.items():
                            if doc_name not in doc_sources:
                                doc_sources[doc_name] = {
                                    "chunk_count": doc_info.get("chunk_count", 0),
                                    "pages": set()
                                }
                        
                        # Display document summary
                        st.markdown("### 📚 Document Summary")
                        
                        summary_data = []
                        for source, source_info in doc_sources.items():
                            summary_data.append({
                                "Document": source,
                                "Pages": len(source_info.get("pages", set())) or 1,  # At least 1 page
                                "Chunks": source_info.get("chunk_count", 0),
                                "ID": source  # Use document name as ID
                            })
                        
                        # Create a pandas DataFrame for display
                        import pandas as pd
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Document selection for deletion
                        st.session_state.selected_docs_for_deletion = st.multiselect(
                            "Select documents to delete:",
                            options=[doc["Document"] for doc in summary_data]
                        )
                        
                        if st.session_state.selected_docs_for_deletion:
                            if st.button("🗑️ Delete Selected Documents", type="primary"):
                                with st.spinner("Deleting documents..."):
                                    for doc_name in st.session_state.selected_docs_for_deletion:
                                        try:
                                            # Find the document ID
                                            doc_id = next((doc["ID"] for doc in summary_data if doc["Document"] == doc_name), None)
                                            if doc_id:
                                                delete_document(doc_id)
                                                st.success(f"Deleted document: {doc_name}")
                                            else:
                                                st.error(f"Could not find document ID for: {doc_name}")
                                        except Exception as e:
                                            st.error(f"Error deleting document: {str(e)}")
                                    
                                    # Reset selection and refresh
                                    st.session_state.selected_docs_for_deletion = None
                                    st.rerun()
                        
                        # Delete all documents button
                        if st.button("🗑️ Delete All Documents"):
                            st.session_state.show_delete_all_warning = True
                        
                        if st.session_state.show_delete_all_warning:
                            st.warning("⚠️ Are you sure you want to delete ALL documents? This action cannot be undone!")
                            
                            confirm_col1, confirm_col2 = st.columns([1, 1])
                            with confirm_col1:
                                if st.button("Yes, delete all", type="primary"):
                                    with st.spinner("Deleting all documents..."):
                                        clear_all_documents()
                                        st.session_state.show_delete_all_warning = False
                                        st.success("All documents have been deleted.")
                                        st.rerun()
                            with confirm_col2:
                                if st.button("Cancel"):
                                    st.session_state.show_delete_all_warning = False
                                    st.rerun()
                    
                    with manage_col2:
                        st.markdown("### 📊 Statistics")
                        
                        # Display document stats
                        st.metric("Total Documents", len(doc_sources))
                        
                        # Calculate total chunks
                        total_chunks = sum(info.get("chunk_count", 0) for _, info in doc_sources.items())
                        st.metric("Total Chunks", total_chunks)
                        
                        # Calculate average chunks per document
                        avg_chunks = total_chunks / len(doc_sources) if doc_sources else 0
                        st.metric("Avg. Chunks per Document", f"{avg_chunks:.1f}")
                        
                        # Calculate total pages
                        total_pages = sum(doc["Pages"] for doc in summary_data)
                        st.metric("Total Pages", total_pages)
                else:
                    st.warning("No documents found in the database. Please upload documents first.")
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")
                if debug_mode:
                    st.error(traceback.format_exc())
        
        # Dropbox Integration Tab
        with dropbox_tab:
            st.subheader("☁️ Dropbox Integration")
            
            # Check if Dropbox is configured
            if not is_dropbox_configured():
                st.warning("⚠️ Dropbox API credentials not configured. Please add them to your .env file.")
                
                # Display required environment variables
                st.markdown("### Required Environment Variables")
                st.code("""
DROPBOX_APP_KEY=your_app_key
DROPBOX_APP_SECRET=your_app_secret
DROPBOX_REFRESH_TOKEN=your_refresh_token
                """)
            else:
                st.success("✅ Dropbox connection configured")
                
                try:
                    # Get Dropbox client
                    dropbox_client = get_dropbox_client()
                    
                    # List folders
                    folders = list_dropbox_folders("")
                    
                    # Folder selection
                    selected_folder = st.selectbox(
                        "Select Dropbox folder:",
                        options=[""] + folders,
                        format_func=lambda x: "Root" if x == "" else x
                    )
                    
                    # List PDF files in selected folder
                    pdf_files = list_dropbox_pdf_files(selected_folder)
                    
                    if pdf_files:
                        st.success(f"Found {len(pdf_files)} PDF files in the selected folder")
                        
                        # Create a nice display of the files
                        file_cols = st.columns(3)
                        
                        for i, file_info in enumerate(pdf_files):
                            col_idx = i % 3
                            with file_cols[col_idx]:
                                st.markdown(f"""
<div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
<strong>📄 {file_info['name']}</strong><br>
Size: {file_info['size'] / 1024:.1f} KB<br>
Modified: {file_info['modified']}
</div>
                                """, unsafe_allow_html=True)
                        
                        # File selection for processing
                        selected_file = st.selectbox(
                            "Select a PDF file to process:",
                            options=[file_info['path'] for file_info in pdf_files],
                            format_func=lambda x: x.split('/')[-1]  # Show only filename
                        )
                        
                        if selected_file and st.button("📥 Process Selected File", type="primary"):
                            with st.spinner("Downloading and processing file from Dropbox..."):
                                try:
                                    # Process the Dropbox file
                                    success = process_dropbox_document(selected_file)
                                    
                                    if success:
                                        st.success(f"✅ Dropbox file processed successfully: {selected_file}")
                                    else:
                                        st.error("Failed to process Dropbox file.")
                                except Exception as e:
                                    st.error(f"Error processing Dropbox file: {str(e)}")
                                    if debug_mode:
                                        st.error(traceback.format_exc())
                    else:
                        st.info("No PDF files found in the selected folder.")
                        
                        # Option to create a new folder
                        new_folder = st.text_input("Create a new folder in Dropbox:")
                        if new_folder and st.button("Create Folder"):
                            try:
                                folder_path = f"{selected_folder}/{new_folder}" if selected_folder else new_folder
                                create_dropbox_folder(folder_path)
                                st.success(f"Folder created: {new_folder}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error creating folder: {str(e)}")
                
                except Exception as e:
                    st.error(f"Error connecting to Dropbox: {str(e)}")
                    if debug_mode:
                        st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error in RAG Explorer: {str(e)}")
        if debug_mode:
            st.error(traceback.format_exc())
