"""
Knowledge Bank tab for MCP Dashboard.
Provides an overview of data sources and database connections.
"""

import os
import streamlit as st
import pandas as pd
import requests
import traceback
import json
import time
from datetime import datetime
from pathlib import Path

from app_database import (
    check_database_status,
    delete_document
)
from app_functions import (
    get_temp_files, 
    clean_temp_files
)

# Environment variables
PG_DB_NAME = os.getenv('POSTGRES_DATABASE', os.getenv('DB_NAME', 'postgres'))
PG_DB_HOST = os.getenv('POSTGRES_HOST', os.getenv('DB_HOST', 'localhost'))
PG_DB_PORT = os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', '5432'))
MYSQL_DB_NAME = os.getenv('MYSQL_DB_NAME', 'defaultdb')
MYSQL_DB_HOST = os.getenv('MYSQL_DB_HOST', 'localhost')
MYSQL_DB_PORT = os.getenv('MYSQL_DB_PORT', '3306')

# Path to temp directory
temp_data_dir = os.path.join(os.getcwd(), "temp_data")

def show_knowledge_tab():
    """
    Display the Knowledge Bank tab that allows users to save and retrieve knowledge snippets
    """
    st.header("Knowledge Bank")
    
    # Get the knowledge bank directory
    knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'knowledge_bank')
    os.makedirs(knowledge_dir, exist_ok=True)
    
    # Knowledge Bank json file
    knowledge_file = os.path.join(knowledge_dir, 'knowledge_entries.json')
    
    # Initialize knowledge entries if file doesn't exist
    if not os.path.exists(knowledge_file):
        with open(knowledge_file, 'w') as f:
            json.dump([], f)
    
    # Load existing knowledge entries
    with open(knowledge_file, 'r') as f:
        knowledge_entries = json.load(f)
    
    # Sidebar for knowledge categories
    with st.sidebar:
        st.header("Knowledge Categories")
        
        # Extract unique categories
        categories = sorted(list(set([entry.get('category', 'Uncategorized') 
                                     for entry in knowledge_entries])))
        
        if not categories:
            categories = ['Uncategorized']
            
        # Add "All" option at the beginning
        filter_options = ['All'] + categories
        
        selected_category = st.selectbox(
            "Filter by category:",
            filter_options,
            index=0
        )
        
        # Search box
        search_query = st.text_input("Search knowledge entries:", "")
        
        # Sort options
        sort_by = st.radio(
            "Sort by:",
            ["Newest first", "Oldest first", "Title (A-Z)", "Title (Z-A)"],
            index=0
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Knowledge Entries")
        
        # Filter entries by category and search query
        filtered_entries = knowledge_entries
        if selected_category != "All":
            filtered_entries = [entry for entry in knowledge_entries 
                              if entry.get('category', 'Uncategorized') == selected_category]
        
        if search_query:
            search_query = search_query.lower()
            filtered_entries = [entry for entry in filtered_entries 
                              if search_query in entry.get('title', '').lower() or 
                                 search_query in entry.get('content', '').lower()]
        
        # Sort entries
        if sort_by == "Newest first":
            filtered_entries = sorted(filtered_entries, 
                                    key=lambda x: x.get('timestamp', 0), 
                                    reverse=True)
        elif sort_by == "Oldest first":
            filtered_entries = sorted(filtered_entries, 
                                    key=lambda x: x.get('timestamp', 0))
        elif sort_by == "Title (A-Z)":
            filtered_entries = sorted(filtered_entries, 
                                    key=lambda x: x.get('title', '').lower())
        elif sort_by == "Title (Z-A)":
            filtered_entries = sorted(filtered_entries, 
                                    key=lambda x: x.get('title', '').lower(), 
                                    reverse=True)
        
        # Display entries
        if not filtered_entries:
            st.info("No knowledge entries found matching your criteria.")
        else:
            for i, entry in enumerate(filtered_entries):
                with st.expander(f"{entry.get('title', 'Untitled')} ({entry.get('category', 'Uncategorized')})"):
                    st.markdown(entry.get('content', 'No content'))
                    
                    # Date display
                    if 'timestamp' in entry:
                        date_str = datetime.fromtimestamp(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                        st.caption(f"Created on: {date_str}")
                    
                    # Delete button
                    if st.button("Delete Entry", key=f"delete_{i}"):
                        knowledge_entries.remove(entry)
                        with open(knowledge_file, 'w') as f:
                            json.dump(knowledge_entries, f)
                        st.success("Entry deleted!")
                        time.sleep(1)
                        st.rerun()
    
    with col2:
        st.subheader("Add New Entry")
        
        # Form for new entry
        with st.form("new_knowledge_entry"):
            new_title = st.text_input("Title:")
            
            # Category dropdown with option to add new
            new_category = st.selectbox(
                "Category:",
                categories + ["+ Add new category"]
            )
            
            # If "Add new category" is selected, show input for new category
            if new_category == "+ Add new category":
                new_category = st.text_input("New category name:")
            
            new_content = st.text_area("Content:", height=200)
            
            submitted = st.form_submit_button("Save Entry")
            
            if submitted and new_title and new_content:
                # Create new entry
                new_entry = {
                    'title': new_title,
                    'category': new_category if new_category else 'Uncategorized',
                    'content': new_content,
                    'timestamp': time.time()
                }
                
                # Add to entries and save
                knowledge_entries.append(new_entry)
                with open(knowledge_file, 'w') as f:
                    json.dump(knowledge_entries, f)
                
                st.success("Knowledge entry saved!")
                time.sleep(1)
                st.rerun()
        
        # Export/Import section
        st.subheader("Export/Import")
        
        # Export to CSV
        if st.button("Export to CSV"):
            if knowledge_entries:
                # Convert to DataFrame
                df = pd.DataFrame(knowledge_entries)
                
                # Convert timestamp to readable date
                if 'timestamp' in df.columns:
                    df['date'] = df['timestamp'].apply(
                        lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')
                    )
                
                # Save to CSV
                export_path = os.path.join(knowledge_dir, 'knowledge_export.csv')
                df.to_csv(export_path, index=False)
                
                # Provide download link
                st.success(f"Knowledge exported to: {export_path}")
            else:
                st.warning("No knowledge entries to export.")
        
        # Import section
        st.subheader("Import from Text")
        
        # Form for importing text
        with st.form("import_knowledge"):
            import_title = st.text_input("Title for imported content:")
            import_category = st.selectbox(
                "Category for imported content:",
                categories,
                key="import_category"
            )
            import_content = st.text_area("Paste content to import:", height=100)
            
            import_submitted = st.form_submit_button("Import Content")
            
            if import_submitted and import_title and import_content:
                # Create new entry
                import_entry = {
                    'title': import_title,
                    'category': import_category,
                    'content': import_content,
                    'timestamp': time.time()
                }
                
                # Add to entries and save
                knowledge_entries.append(import_entry)
                with open(knowledge_file, 'w') as f:
                    json.dump(knowledge_entries, f)
                
                st.success("Content imported successfully!")
                time.sleep(1)
                st.rerun()
    
    # Database Connection Pool Statistics
    st.subheader("🔌 Database Connection Status")
    try:
        # Display database connection status
        conn_cols = st.columns(2)
        with conn_cols[0]:
            st.markdown("**PostgreSQL Connection**")
            postgres_status = check_database_status("postgres")
            status_text = "ONLINE" if postgres_status else "OFFLINE"
            status_color = "green" if postgres_status else "red"
            st.markdown(f"Status: <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
            
        with conn_cols[1]:
            st.markdown("**MySQL Connection**")
            mysql_status = check_database_status("mysql")
            status_text = "ONLINE" if mysql_status else "OFFLINE"
            status_color = "green" if mysql_status else "red"
            st.markdown(f"Status: <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error checking database connections: {str(e)}") 