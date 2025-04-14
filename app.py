"""
app_old.py - MCP Streamlit Application with direct database connections

Note: This version uses direct database connections instead of MCP servers. 
References to POSTGRES_SERVER_URL and MYSQL_SERVER_URL in the Knowledge Bank tab still
need to be updated to use direct connections.
"""

import os
import streamlit as st
import requests
from dotenv import load_dotenv
import pandas as pd
import polars as pl
import json
import subprocess
import time
import tempfile
from datetime import datetime
import uuid
import socket
import sys
import argparse
import logging
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasAIOpenAI
import pandasai as pai
from PIL import Image
import base64
import numpy as np
import io
from io import StringIO, BytesIO
from langchain.schema import Document
import pymysql
import openai
from openai import OpenAI
import psycopg2
from pathlib import Path

# Import app modules
from app_documents import (
    process_pdf_file, 
    delete_document_from_db
)

from app_vector import (
    create_vector_store,
    get_vector_store
)

from app_retrieval import (
    hybrid_retriever,
    DocumentCollection
)

from app_database import (
    get_connection_string,
    make_json_safe,
    initialize_database,
    get_db_connection, 
    connect_mysql,
    check_database_status, 
    execute_postgres_query, 
    execute_mysql_query,
    sanitize_table_name
)

from app_dropbox import (
    initialize_dropbox_client,
    is_dropbox_configured,
    get_file_from_dropbox,
    list_dropbox_folders,
    list_dropbox_pdf_files,
    create_file_like_object,
    set_dropbox_on_demand,
    get_dropbox_usage_stats,
    get_dropbox_client
)

from app_rag import (
    implement_parent_document_retriever,
    fetch_parent_context
)

from app_search import (
    natural_language_sql_search,
    sql_keyword_search,
    simple_sql_search
)

from app_functions import (
    format_retrieval_context,
    create_prompt_with_context,
    extract_and_cite_answer,
    process_user_query,
    format_metrics_for_streamlit
)

from app_functions import (
    get_temp_files,
    clean_temp_files,
    get_answer_from_documents
)

# Import tab modules
from tabs.tab1_postgres import show_postgres_tab
from tabs.tab2_mysql import show_mysql_tab
from tabs.tab3_rag import show_rag_tab
from tabs.tab4_analysis import show_analysis_tab
from tabs.tab5_knowledge import show_knowledge_tab

# Connect to Postgres wrapper function for backward compatibility
def connect_postgres():
    return get_db_connection()

# Load environment variables
load_dotenv()

# Database configurations from environment variables
PG_DB_NAME = os.getenv('POSTGRES_DATABASE', os.getenv('DB_NAME', 'postgres'))
PG_DB_USER = os.getenv('POSTGRES_USER', os.getenv('DB_USER', 'postgres'))
PG_DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', os.getenv('DB_PASSWORD', ''))
PG_DB_HOST = os.getenv('POSTGRES_HOST', os.getenv('DB_HOST', 'localhost'))
PG_DB_PORT = os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', '5432'))

MYSQL_DB_NAME = os.getenv('MYSQL_DB_NAME', 'defaultdb')
MYSQL_DB_USER = os.getenv('MYSQL_DB_USER', 'root')
MYSQL_DB_PASSWORD = os.getenv('MYSQL_DB_PASSWORD', '')
MYSQL_DB_HOST = os.getenv('MYSQL_DB_HOST', 'localhost')
MYSQL_DB_PORT = os.getenv('MYSQL_DB_PORT', '3306')

# Global variable to store temporary data directory
temp_data_dir = os.path.join(os.getcwd(), "temp_data")

# Create temp directory if it doesn't exist
if not os.path.exists(temp_data_dir):
    os.makedirs(temp_data_dir)

# Initialize settings in session state if not already present
if 'use_polars' not in st.session_state:
    st.session_state.use_polars = True  # Default to using Polars for better performance
if 'analysis_backend' not in st.session_state:
    st.session_state.analysis_backend = "openrouter"  # Default to OpenRouter
if 'current_temp_file' not in st.session_state:
    st.session_state.current_temp_file = None  # Track the current temp file

# Set Streamlit page config (must be the first command)
st.set_page_config(
    page_title="MCP Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Title and server status
st.title("Athletics Dashboard")
st.markdown("### Powered by OpenRouter & DeepSeek Chat v3 Model")
st.info("Using direct database connections instead of MCP servers")

# Main application
def main():
    # Create tabs for different functionality
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "PostgreSQL Explorer", "MySQL Explorer", 
        "RAG Explorer", "Data Analysis", "Knowledge Bank"
    ])
    
    # Tab 1: PostgreSQL Explorer
    with tab1:
        show_postgres_tab()
    
    # Tab 2: MySQL Explorer
    with tab2:
        show_mysql_tab()
    
    # Tab 3: RAG Explorer 
    with tab3:
        show_rag_tab()
    
    # Tab 4: Data Analysis
    with tab4:
        show_analysis_tab()
    
    # Tab 5: Knowledge Bank
    with tab5:
        show_knowledge_tab()

# Footer and cleanup
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, FastAPI, PyGithub, and OpenRouter")

def cleanup():
    """Cleanup function to run when the app is shutting down"""
    logging.info("Running cleanup")
    clean_temp_files(keep_current=False)
    logging.info("Cleanup completed")

st.session_state.cleanup = cleanup

if __name__ == "__main__":
    main()
