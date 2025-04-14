import os
import logging
import requests
import json
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get OpenRouter API key from environment variables
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if OPENROUTER_API_KEY:
    logger.info("OpenRouter API key found")
else:
    logger.warning("No OpenRouter API key found in environment variables")

def nl_to_sql(query: str, schema_info: str = None, db_type: str = "postgresql"):
    """
    Convert natural language query to SQL using OpenRouter API with DeepSeek model
    
    Parameters:
        query: The natural language query
        schema_info: Database schema information (optional)
        db_type: Database type, either "postgresql" or "mysql" (default: "postgresql")
    """
    try:
        if not OPENROUTER_API_KEY:
            logger.error("OpenRouter API key not configured")
            return {
                "sql": f"-- OpenRouter API key not configured",
                "explanation": "OpenRouter API key is missing from environment variables"
            }
        
        # Prepare the system message with schema information if available
        if db_type.lower() == "mysql":
            system_content = """You are a SQL expert. Convert natural language queries to MySQL SQL statements.
            
            IMPORTANT GUIDELINES:
            1. Generate valid MySQL syntax
            2. MySQL identifiers are case-sensitive on some systems, so use backticks around table and column names
            3. Use EXACTLY the table and column names as provided in the schema information - do not change case
            4. Return valid MySQL syntax that can be executed directly
            5. If the user mentions a table name that doesn't match the exact case of tables in the schema, use the exact case from the schema instead
            """
        else:  # postgresql
            system_content = """You are a SQL expert. Convert natural language queries to PostgreSQL SQL statements.
            
            IMPORTANT GUIDELINES:
            1. ALWAYS use double quotes around table and column names to preserve their case sensitivity
            2. PostgreSQL identifiers are case-sensitive when quoted, so "tablename" is different from "TableName"
            3. Use EXACTLY the table and column names as provided in the schema information - do not change case
            4. Return valid PostgreSQL syntax that can be executed directly
            5. If the user mentions a table name that doesn't match the exact case of tables in the schema, use the exact case from the schema instead
            """
        
        if schema_info:
            system_content += f"\n\nDatabase schema information:\n{schema_info}"
        
        logger.info(f"Making OpenRouter API call for NL query: {query} (Database: {db_type})")
        
        # Direct HTTP request to OpenRouter API
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://mcp-learning.example.com",
            "Content-Type": "application/json"
        }
        
        # Craft user prompt based on database type
        if db_type.lower() == "mysql":
            user_prompt = f"Convert this natural language query to MySQL SQL, using EXACTLY the table and column names provided in the schema: '{query}'. Remember to use backticks around ALL table and column names."
        else:  # postgresql
            user_prompt = f"Convert this natural language query to PostgreSQL SQL, using EXACTLY the table and column names provided in the schema: '{query}'. Remember to use double quotes around ALL table and column names."
        
        data = {
            "model": "deepseek/deepseek-r1:free",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }
        
        # Send request
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        # Check if request was successful
        if response.status_code == 200:
            # Parse the response
            result_json = response.json()
            logger.info("OpenRouter API response received successfully")
            
            # Extract the generated SQL from the response
            sql_response = result_json['choices'][0]['message']['content'].strip()
            logger.info(f"Raw SQL response: {sql_response}")
            
            # Try to extract SQL code from markdown code blocks
            if "```sql" in sql_response or "```" in sql_response:
                # Handle both ```sql and plain ``` code blocks
                # First try ```sql specific format
                if "```sql" in sql_response:
                    parts = sql_response.split("```sql")
                    if len(parts) > 1:
                        sql_code = parts[1].split("```")[0].strip()
                        explanation = sql_response.replace(f"```sql{parts[1].split('```')[0]}```", "").strip()
                    else:
                        sql_code = sql_response
                        explanation = ""
                # Then try just ``` format
                elif "```" in sql_response:
                    parts = sql_response.split("```")
                    if len(parts) > 1:
                        # The SQL should be in the second part (index 1)
                        sql_code = parts[1].strip()
                        # Remove any language identifier if present
                        if sql_code.startswith("sql"):
                            sql_code = sql_code[3:].strip()
                        explanation = sql_response.replace(f"```{parts[1]}```", "").strip()
                    else:
                        sql_code = sql_response
                        explanation = ""
            else:
                # Try to identify SQL by keywords
                sql_keywords = ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "JOIN", "LIMIT"]
                lines = sql_response.splitlines()
                sql_lines = []
                explanation_lines = []
                
                for line in lines:
                    if any(keyword in line.upper() for keyword in sql_keywords):
                        sql_lines.append(line)
                    else:
                        explanation_lines.append(line)
                
                sql_code = "\n".join(sql_lines).strip()
                explanation = "\n".join(explanation_lines).strip()
                
                if not sql_code:
                    sql_code = sql_response
                    explanation = ""
            
            # Final cleanup to ensure no markdown or sql tags remain
            sql_code = sql_code.replace("```sql", "").replace("```", "").strip()
            
            # Add appropriate quotes around table names if they aren't already quoted
            import re
            if db_type.lower() == "mysql":
                # For MySQL, use backticks
                table_pattern = r'FROM\s+([A-Za-z0-9_]+)'
                sql_code = re.sub(table_pattern, r'FROM `\1`', sql_code)
                
                # Also handle JOIN statements
                join_pattern = r'JOIN\s+([A-Za-z0-9_]+)'
                sql_code = re.sub(join_pattern, r'JOIN `\1`', sql_code)
            else:
                # For PostgreSQL, use double quotes
                table_pattern = r'FROM\s+([A-Za-z0-9_]+)'
                sql_code = re.sub(table_pattern, r'FROM "\1"', sql_code)
                
                # Also handle JOIN statements
                join_pattern = r'JOIN\s+([A-Za-z0-9_]+)'
                sql_code = re.sub(join_pattern, r'JOIN "\1"', sql_code)
            
            result = {
                "sql": sql_code,
                "explanation": explanation
            }
        else:
            # Log error response
            logger.error(f"OpenRouter API request failed with status {response.status_code}: {response.text}")
            result = {
                "sql": "-- Error occurred",
                "explanation": f"OpenRouter API request failed: {response.status_code} - {response.text}"
            }
        
        logger.info(f"Processed result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error calling OpenRouter API: {str(e)}", exc_info=True)
        return {
            "sql": "-- Error occurred",
            "explanation": f"Error calling OpenRouter API: {str(e)}"
        }

def github_analyze(repo_name: str, task: str):
    """Use OpenRouter API to analyze GitHub repository data"""
    try:
        if not OPENROUTER_API_KEY:
            logger.error("OpenRouter API key not configured")
            return {
                "error": "OpenRouter API key not configured"
            }
        
        logger.info(f"Making OpenRouter API call for GitHub analysis: {repo_name}")
        
        # Make the API call to OpenRouter directly using requests
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://mcp-learning.example.com",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [
                {"role": "system", "content": "You are a GitHub repository expert. Analyze repositories and provide insights."},
                {"role": "user", "content": f"Repository: {repo_name}\n\nTask: {task}"}
            ],
            "temperature": 0.5
        }
        
        # Use the correct OpenRouter API endpoint with retry logic
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30  # Add timeout to avoid hanging forever
                )
                
                # Check if rate limited
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limited by OpenRouter API. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                
                # If we got a response, break the loop
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Request failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
        
        # Check if request was successful
        if response.status_code == 200:
            # Extract the analysis
            result_json = response.json()
            analysis = result_json['choices'][0]['message']['content'].strip()
            logger.info("GitHub analysis completed successfully")
            
            return {
                "analysis": analysis
            }
        else:
            # Log error response
            logger.error(f"OpenRouter API request failed with status {response.status_code}: {response.text}")
            return {
                "error": f"OpenRouter API request failed: {response.status_code} - {response.text}"
            }
    except Exception as e:
        logger.error(f"Error calling OpenRouter API: {str(e)}", exc_info=True)
        return {
            "error": f"Error calling OpenRouter API: {str(e)}"
        } 