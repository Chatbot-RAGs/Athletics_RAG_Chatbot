"""
MySQL Explorer tab for Aspire Academy Athletics Dashboard.
Provides MySQL database query interface, SQL generation, and table operations.
"""

import os
import streamlit as st
import pandas as pd
import traceback
import openai
import requests

from app_database import (
    connect_mysql,
    check_database_status,
    execute_mysql_query
)
from app_functions import save_dataframe

# We don't need to hardcode these variables - they're already accessed through app_database functions
# and os.getenv() will be used when needed

def show_mysql_tab():
    """Display the MySQL Explorer tab content."""
    st.header("MySQL Explorer")
    
    # Get database name from environment for display purposes
    mysql_db_name = os.getenv('MYSQL_DB_NAME', 'defaultdb')
    
    # List tables
    if st.button("List Tables", key="mysql_list_tables"):
        try:
            conn = connect_mysql()  # No need to pass database name, it's handled in the function
            if conn:
                cursor = conn.cursor()
                cursor.execute(f"SHOW TABLES FROM `{mysql_db_name}`")
                tables = [row[0] for row in cursor.fetchall()]
                cursor.close()
                conn.close()
                if tables:
                    st.write(f"### Available Tables in {mysql_db_name}")
                    for table in tables:
                        st.write(f"- {table}")
                else:
                    st.info(f"No tables found in database '{mysql_db_name}'.")
            else:
                st.error(f"Could not connect to MySQL database '{mysql_db_name}'.")
        except Exception as e:
            st.error(f"Error connecting to MySQL database: {str(e)}")
            st.error(traceback.format_exc())

    # Database query interface
    st.subheader("Query Database")
    mysql_query = st.text_area("Enter your MySQL query", key="mysql_query_area")
    save_mysql_for_analysis = st.checkbox("Save results for analysis", value=True, key="save_mysql_results")

    if mysql_query and st.button("Execute MySQL Query", key="execute_mysql"):
        try:
            with st.spinner("Executing MySQL query..."):
                st.info("Executing your MySQL query...")
                st.code(mysql_query, language="sql")
            
            # Execute query using the centralized function
            result = execute_mysql_query(mysql_query)  # No need to pass database name
            
            # Display results
            if result["success"]:
                if len(result["data"]) > 0:
                    st.success(f"Query returned {len(result['data'])} rows")
                    df = result["dataframe"]
                    
                    # Show column information
                    st.write("### Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count().values,
                        'First Value': [str(df[col].iloc[0]) if not df[col].isna().all() and len(df) > 0 else "NULL" for col in df.columns]
                    })
                    st.dataframe(col_info)
                    
                    # Store dataframe in session state
                    if save_mysql_for_analysis:
                        # Save to session state for later analysis
                        st.session_state.last_query_df = df
                        # Save to disk as a single temp file
                        filepath = save_dataframe(df, "mysql_query", reuse_current=False)
                        st.session_state.last_query_filepath = filepath
                        # Update the current temp file reference
                        st.session_state.current_temp_file = filepath
                        # Limit display size to prevent page freezing
                        row_count_display = len(df)
                        st.dataframe(df.head(5))
                        st.info(f"Dataset with {row_count_display} rows is saved and available for analysis in the Data Analysis tab.")
                        
                        # Add option to download full result
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download full results as CSV",
                            csv,
                            "mysql_query_results.csv",
                            "text/csv",
                            key='download-mysql-csv'
                        )
                    else:
                        row_count_display = len(df)
                        st.dataframe(df.head(5))
                        st.info(f"Dataset with {row_count_display} rows is available in memory.")
                else:
                    st.info("Query executed successfully, but returned no results.")
            elif 'error' in result:
                st.error(result['error'])
            else:
                st.json(result)
                
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            st.error(traceback.format_exc())

    # Natural language query with OpenRouter
    st.subheader("Natural Language Query")
    st.markdown("Powered by DeepSeek Chat v3 model via OpenRouter")
    
    mysql_nl_query = st.text_input("Describe what you want to query", key="mysql_nl_query")
    should_execute_mysql = st.checkbox("Execute the generated SQL", value=True, key="execute_mysql_nl_sql")
    save_mysql_nl_results = st.checkbox("Save results for analysis", value=True, key="save_mysql_nl_results")
    
    if mysql_nl_query and st.button("Generate & Execute MySQL SQL", key="generate_mysql_sql"):
        with st.spinner("Generating MySQL SQL with AI..."):
            try:
                # Generate SQL from natural language using OpenRouter API
                
                # OpenRouter API key
                openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                if not openrouter_api_key:
                    st.error("OpenRouter API key not found. Please add it to your .env file as OPENROUTER_API_KEY.")
                    continue_execution = False
                else:
                    continue_execution = True
                
                if continue_execution:
                    try:
                        conn = connect_mysql()  # No need to pass database name
                        mysql_db_name = os.getenv('MYSQL_DB_NAME', 'defaultdb')  # Get DB name for context
                        if conn:
                            cursor = conn.cursor()
                            cursor.execute(f"SHOW TABLES FROM `{mysql_db_name}`")
                            tables = [row[0] for row in cursor.fetchall()]
                            schema_info = []
                            for table in tables[:10]:
                                cursor.execute(f"DESCRIBE `{mysql_db_name}`.`{table}`")
                                columns = cursor.fetchall()
                                schema_info.append(f"Table: {table}")
                                schema_info.append("Columns: " + ", ".join([f"{col[0]} ({col[1]})" for col in columns]))
                            cursor.close()
                            conn.close()
                            schema_context = "\n".join(schema_info)
                        else:
                            schema_context = f"Could not connect to database '{mysql_db_name}' to retrieve schema information."
                    except Exception as e:
                        schema_context = f"Error getting schema: {str(e)}"

                    prompt = f"""
                    Convert this natural language query to valid SQL.
                    Database: MySQL (database name: {mysql_db_name})

                    Database Schema:
                    {schema_context}

                    IMPORTANT: 
                    1. ONLY use tables and columns that are explicitly listed in the schema above.
                    2. DO NOT reference any tables or columns that are not listed in the schema.
                    3. Include the database name in your queries, but DO NOT use backticks.
                    4. For example, use: SELECT * FROM {mysql_db_name}.table_name instead of SELECT * FROM `{mysql_db_name}`.`table_name`
                    5. If you cannot find appropriate tables for the query, use a simple query on one of the available tables.
                    6. For athletics data, be flexible with event names:
                       - For '100m', search using '%100%' not just '100 Metres'
                       - For '800m', search using '%800%' not just '800 Metres'
                       - Use broader patterns to catch all variations
                    7. Use LIKE with wildcards for text searches to improve matching, e.g., WHERE Event LIKE '%800%'
                    8. DO NOT use backticks (`) around table or column names as they can cause issues.

                    Natural language query: {mysql_nl_query}

                    Respond with ONLY the SQL query, nothing else. Make sure it's syntactically correct MySQL.
                    """
                    
                    # Import the specialized MySQL LLM response function
                    from app_llm import get_mysql_llm_response
                    
                    # Use the specialized function with DeepSeek Chat v3 model
                    sql_query = get_mysql_llm_response(prompt)
                    if "```sql" in sql_query or "```" in sql_query:
                        if "```sql" in sql_query:
                            parts = sql_query.split("```sql")
                            if len(parts) > 1:
                                sql_query = parts[1].split("```")[0].strip()
                        elif "```" in sql_query:
                            parts = sql_query.split("```")
                            if len(parts) > 1:
                                sql_query = parts[1].strip()
                                if sql_query.startswith("sql"):
                                    sql_query = sql_query[3:].strip()
                    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
                    st.write("### Generated SQL")
                    st.code(sql_query, language="sql")
                    
                    # Check if the query might reference non-existent tables
                    # This is a simple check that looks for table names in the query
                    # that aren't in the list of tables we got from the database
                    if tables:
                        # Extract potential table names from the query
                        # This is a simple approach and might not catch all cases
                        query_words = sql_query.replace(';', ' ').replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
                        potential_tables = [word for word in query_words if word.lower() not in 
                                          ['select', 'from', 'where', 'group', 'by', 'having', 'order', 'limit', 
                                           'join', 'inner', 'outer', 'left', 'right', 'on', 'as', 'and', 'or', 
                                           'not', 'in', 'between', 'is', 'null', 'like', 'distinct', 'count', 
                                           'sum', 'avg', 'min', 'max', 'case', 'when', 'then', 'else', 'end',
                                           'union', 'all', 'insert', 'update', 'delete', 'create', 'alter', 'drop',
                                           'table', 'view', 'index', 'primary', 'key', 'foreign', 'references',
                                           'default', 'constraint', 'unique', 'check', 'column', 'database', 'schema']]
                        
                        # Check if any potential table name is not in the list of tables
                        unknown_tables = [table for table in potential_tables if table not in tables and '.' not in table]
                        if unknown_tables:
                            st.warning(f"⚠️ The generated SQL query might reference tables that don't exist: {', '.join(unknown_tables)}")
                            st.info("Available tables: " + ", ".join(tables))
                            st.info("Consider clicking 'List Tables' to see all available tables, then try again with a more specific query.")
                    
                    if should_execute_mysql:
                        result = execute_mysql_query(sql_query)  # No need to pass database name
                        if result["success"]:
                            if len(result["data"]) > 0:
                                st.success(f"Query returned {len(result['data'])} rows")
                                if "dataframe" in result and isinstance(result["dataframe"], pd.DataFrame):
                                    df = result["dataframe"]
                                elif isinstance(result["data"], list):
                                    try:
                                        if result["data"] and isinstance(result["data"][0], dict):
                                            df = pd.DataFrame(result["data"])
                                        elif result["data"] and isinstance(result["data"][0], list):
                                            if "column_names" in result and result["column_names"]:
                                                df = pd.DataFrame(result["data"], columns=result["column_names"])
                                            else:
                                                df = pd.DataFrame(result["data"])
                                                if all(isinstance(col, int) for col in df.columns):
                                                    df.columns = [f"Column_{i}" for i in range(len(df.columns))]
                                        else:
                                            df = pd.DataFrame(result["data"])
                                    except Exception as convert_error:
                                        st.error(f"Error converting data to DataFrame: {str(convert_error)}")
                                        st.json(result["data"][:5])
                                        st.stop()
                                else:
                                    st.warning("Unexpected data format returned")
                                    st.json(result["data"][:10])
                                    st.stop()

                                st.write("### Column Information")
                                try:
                                    col_info = pd.DataFrame({
                                        'Column': df.columns,
                                        'Type': [str(dtype) for dtype in df.dtypes],
                                        'Non-Null Count': df.count().values,
                                        'First Value': [str(df[col].iloc[0]) if not df[col].isna().all() and len(df) > 0 else "NULL" for col in df.columns]
                                    })
                                    st.dataframe(col_info)
                                except Exception as e:
                                    st.error(f"Error creating column info: {str(e)}")
                                    st.write("Displaying data without column information")
                                st.dataframe(df.head(10))
                                if save_mysql_nl_results:
                                    st.session_state.last_query_df = df
                                    filepath = save_dataframe(df, "mysql_nl_query", reuse_current=False)
                                    st.session_state.last_query_filepath = filepath
                                    st.session_state.current_temp_file = filepath
                                    st.info(f"Dataset with {len(df)} rows is saved and available for analysis in the Data Analysis tab.")
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        "Download full results as CSV",
                                        csv,
                                        "mysql_nl_query_results.csv",
                                        "text/csv",
                                        key='download-mysql-nl-csv'
                                    )
                                else:
                                    st.info(f"Dataset with {len(df)} rows is available in memory.")
                            else:
                                st.info("Query executed successfully, but returned no results.")
                        elif 'error' in result:
                            st.error(result['error'])
                        else:
                            st.json(result)
            except Exception as e:
                st.error(f"Error in natural language query: {str(e)}")
                st.error(traceback.format_exc())
