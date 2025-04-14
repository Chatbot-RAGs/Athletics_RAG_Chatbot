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
                    You are a SQL query generator. Convert this natural language query to valid SQL.
                    Database: MySQL (database name: {mysql_db_name})

                    Database Schema:
                    {schema_context}

                    Natural language query: {mysql_nl_query}

                    Respond with ONLY the SQL query, nothing else. Make sure it's syntactically correct MySQL.
                    """
                    client = openai.OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=openrouter_api_key,
                    )
                    response = client.chat.completions.create(
                        model="deepseek/deepseek-chat-v3-0324:free",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                    )
                    sql_query = response.choices[0].message.content.strip()
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