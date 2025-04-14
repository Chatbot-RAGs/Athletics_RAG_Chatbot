"""
PostgreSQL tab for MCP Dashboard.
Provides database query interface, SQL generation, and table schema viewing.
"""

import os
import streamlit as st
import pandas as pd
import requests
import traceback

from app_database import (
    get_db_connection,
    check_database_status,
    execute_postgres_query
)
from app_functions import save_dataframe

# Environment variables
# PG_DB_NAME = os.getenv('POSTGRES_DATABASE', os.getenv('DB_NAME', 'postgres'))
# PG_DB_USER = os.getenv('POSTGRES_USER', os.getenv('DB_USER', 'postgres'))
# PG_DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', os.getenv('DB_PASSWORD', ''))
# PG_DB_HOST = os.getenv('POSTGRES_HOST', os.getenv('DB_HOST', 'localhost'))
# PG_DB_PORT = os.getenv('POSTGRES_PORT', os.getenv('DB_PORT', '5432'))
MYSQL_DB_NAME = os.getenv('MYSQL_DB_NAME', 'defaultdb')
MYSQL_DB_HOST = os.getenv('MYSQL_DB_HOST', 'localhost')
MYSQL_DB_PORT = os.getenv('MYSQL_DB_PORT', '3306')

# Connect to Postgres wrapper function for backward compatibility
def connect_postgres():
    return get_db_connection()

def show_postgres_tab():
    """Display the PostgreSQL tab content."""
    st.header("Database Query")
    
    # List tables using direct database connection
    if st.button("List Tables"):
        try:
            conn = connect_postgres()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                tables = [row[0] for row in cursor.fetchall()]
                cursor.close()
                conn.close()
                # Display tables
                if tables:
                    st.write("### Available Tables")
                    for table in tables:
                        st.write(f"- {table}")
                else:
                    st.info("No tables found in the database.")
            else:
                st.error("Could not connect to PostgreSQL database.")
        except Exception as e:
            st.error(f"Error connecting to PostgreSQL database: {str(e)}")
            st.error(traceback.format_exc())

    # Database query interface
    st.subheader("Query Database")
    query = st.text_area("Enter your SQL query")
    save_for_analysis = st.checkbox("Save results for analysis", value=True, key="save_sql_results")

    if query and st.button("Execute Query"):
        try:
            with st.spinner("Executing query..."):
                st.code(query, language="sql")
            result = execute_postgres_query(query)
            if result["success"]:
                if len(result["data"]) > 0:
                    st.success(f"Query returned {len(result['data'])} rows")
                    # Ensure we have a proper dataframe
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

                    # Display column information if available
                    if isinstance(df, pd.DataFrame):
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
                    # Store dataframe in session state
                    if save_for_analysis:
                        st.session_state.last_query_df = df
                        filepath = save_dataframe(df, "sql_query", reuse_current=False)
                        st.session_state.last_query_filepath = filepath
                        st.session_state.current_temp_file = filepath
                        st.info(f"Dataset with {len(df)} rows is saved and available for analysis in the Data Analysis tab.")
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download full results as CSV",
                            csv,
                            "query_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    else:
                        st.info(f"Dataset with {len(df)} rows is available in memory.")
                else:
                    st.info("Query executed successfully, but returned no results.")
            elif 'error' in result:
                st.error(result['error'])
                if "relation" in result['error'] and "does not exist" in result['error']:
                    st.warning("The table you're trying to query doesn't exist. Let's see what tables are available:")
                    try:
                        conn = connect_postgres()
                        if conn:
                            cursor = conn.cursor()
                            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
                            available_tables = [row[0] for row in cursor.fetchall()]
                            cursor.close()
                            conn.close()
                            if available_tables:
                                st.info("### Available Tables")
                                st.write("Try querying one of these tables instead:")
                                for table in available_tables:
                                    st.write(f"- `{table}`")
                        else:
                            st.info("No tables found in the database.")
                    except Exception as e:
                        st.error(f"Error fetching available tables: {str(e)}")
                else:
                    st.json(result)
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            st.error(traceback.format_exc())

    # Natural language query with OpenRouter
    st.subheader("Natural Language Query")
    st.markdown("Powered by DeepSeek Chat v3 model via OpenRouter")
    nl_query = st.text_input("Describe what you want to query")
    execute_query_flag = st.checkbox("Execute the generated SQL", value=True, key="execute_nl_sql")
    save_nl_results = st.checkbox("Save results for analysis", value=True, key="save_nl_results")

    if nl_query and st.button("Generate & Execute SQL"):
        with st.spinner("Generating SQL with AI..."):
            try:
                openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
                if not openrouter_api_key:
                    st.error("OpenRouter API key not found. Please add it to your .env file as OPENROUTER_API_KEY.")
                else:
                    try:
                        conn = connect_postgres()
                        if conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                SELECT table_name 
                                FROM information_schema.tables 
                                WHERE table_schema = 'public'
                                ORDER BY table_name
                            """)
                            tables = [row[0] for row in cursor.fetchall()]
                            schema_info = []
                            for table in tables[:10]:
                                cursor.execute(f"""
                                    SELECT column_name, data_type, is_nullable
                                    FROM information_schema.columns
                                    WHERE table_name = '{table}'
                                    ORDER BY ordinal_position
                                """)
                                columns = cursor.fetchall()
                                schema_info.append(f"Table: {table}")
                                schema_info.append("Columns: " + ", ".join([f"{col[0]} ({col[1]})" for col in columns]))
                            cursor.close()
                            conn.close()
                            schema_context = "\n".join(schema_info)
                        else:
                            schema_context = "Could not connect to database to retrieve schema information."
                    except Exception as e:
                        schema_context = f"Error getting schema: {str(e)}"

                    prompt = f"""
                    You are a SQL query generator for PostgreSQL. Convert this natural language query to valid SQL.
                    
                    Database Schema Information:
                    {schema_context}
                    
                    User Query: {nl_query}
                    
                    Instructions:
                    1. Generate only valid PostgreSQL SQL.
                    2. Only output the SQL query, nothing else.
                    3. Do not include any explanation or markdown syntax.
                    4. If you're unsure about table or column names, use the schema information provided.
                    5. Keep the query focused and efficient.
                    
                    SQL Query:
                    """
                    data = {
                        "model": "deepseek/deepseek-chat-v3-0324:free",
                        "messages": [
                            {"role": "system", "content": "You are a PostgreSQL SQL query generator. Generate only valid SQL without explanation."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1
                    }
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openrouter_api_key}",
                            "HTTP-Referer": "https://mcp-learning.example.com",
                            "Content-Type": "application/json"
                        },
                        json=data,
                        timeout=30
                    )
                    if response.status_code == 200:
                        result = response.json()
                        generated_sql = result['choices'][0]['message']['content'].strip()
                        st.write("### Generated SQL")
                        st.code(generated_sql, language='sql')
                        if execute_query_flag:
                            st.write("### Query Results")
                            with st.spinner("Executing SQL..."):
                                result = execute_postgres_query(generated_sql)
                                if result["success"]:
                                    if len(result["data"]) > 0:
                                        df = result["dataframe"]
                                        if save_nl_results:
                                            st.session_state.last_query_df = df
                                            filepath = save_dataframe(df, "nl_query", reuse_current=False)
                                            st.session_state.last_query_filepath = filepath
                                            st.session_state.current_temp_file = filepath
                                            st.dataframe(df.head(5))
                                            st.info(f"Dataset with {len(df)} rows is saved and available for analysis in the Data Analysis tab.")
                                            csv = df.to_csv(index=False).encode('utf-8')
                                            st.download_button(
                                                "Download full results as CSV",
                                                csv,
                                                "query_results.csv",
                                                "text/csv",
                                                key='download-csv-nl'
                                            )
                                        else:
                                            st.dataframe(df.head(5))
                                            st.info(f"Dataset with {len(df)} rows is available in memory.")
                                    else:
                                        st.info("Query executed successfully, but returned no results.")
                                else:
                                    st.error(f"Error executing SQL: {result.get('error', 'Unknown error')}")
                    else:
                        st.error(f"Error from OpenRouter API: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.error(traceback.format_exc())

    # Get table schema section
    # st.subheader("View Table Schema")
    # schema_table_name = st.text_input("Enter table name to view schema", key="postgres_schema_table")
    # if schema_table_name and st.button("View Schema", key="postgres_view_schema"):
    #     try:
    #         conn = connect_postgres()
    #         if conn:
    #             cursor = conn.cursor()
    #             cursor.execute("""
    #                 SELECT column_name, data_type, is_nullable 
    #                 FROM information_schema.columns 
    #                 WHERE table_name = %s
    #                 ORDER BY ordinal_position
    #             """, (schema_table_name,))
    #             columns = []
    #             for row in cursor.fetchall():
    #                 columns.append({
    #                     "Column Name": row[0],
    #                     "Data Type": row[1],
    #                     "Is Nullable": row[2]
    #                 })
    #             cursor.close()
    #             conn.close()
    #             if columns:
    #                 st.write(f"### Schema for {schema_table_name}")
    #                 schema_df = pd.DataFrame(columns)
    #                 st.dataframe(schema_df)
    #             else:
    #                 st.error(f"Table '{schema_table_name}' not found or has no columns.")
    #         else:
    #             st.error("Could not connect to PostgreSQL database.")
    #     except Exception as e:
    #         st.error(f"Error getting schema: {str(e)}")
    #         st.error(traceback.format_exc())

    # Move server status to sidebar bottom
    with st.sidebar:
        # Add some spacing to push status to bottom
        st.markdown("<br>" * 5, unsafe_allow_html=True)
        st.markdown("---")
        
        # Compact database status indicators
        postgres_status = check_database_status("postgres")
        mysql_status = check_database_status("mysql")
        
        # Use colored circles with database names on the same line
        pg_indicator = "🟢" if postgres_status else "🔴"
        mysql_indicator = "🟢" if mysql_status else "🔴"
        
        st.markdown(f"{pg_indicator} **PostgreSQL**")
        st.markdown(f"{mysql_indicator} **MySQL**") 