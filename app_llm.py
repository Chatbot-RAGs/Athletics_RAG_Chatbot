"""
app_llm.py

This module handles interactions with large language models.
Provides:
- OpenAI client initialization using OpenRouter as a proxy
- Question answering functionality based on provided context
- Uses DeepSeek Chat v3 model (free) via OpenRouter
- Structured prompting for comprehensive and engaging responses
"""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_answer(question, context, retrieved_docs=None):
    """
    Get a concise, factual answer from a language model based on the question and context.
    Used primarily for RAG (Retrieval Augmented Generation) in the Document Q&A tab.

    This function is designed to provide direct, factual responses with clear source citations
    when answering questions about documents. It uses a lower temperature setting to ensure
    precise, accurate answers rather than creative storytelling.

    Args:
        question (str): The user's question about documents.
        context (str): The relevant context retrieved from documents.
        retrieved_docs (list, optional): List of retrieved document objects with metadata.

    Returns:
        tuple: (answer, sources_text) where answer is the generated response and
               sources_text is formatted text listing the sources used.
    """
    try:
        # Check for OpenRouter API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logging.error("OPENROUTER_API_KEY not found in environment variables")
            return ("I'm sorry, but I can't answer that question because my API key is missing.", "")
        
        # Initialize OpenAI client with OpenRouter base URL
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Detect if any of the retrieved documents indicate that there are tables involved.
        has_tables = False
        table_info = []
        if retrieved_docs:
            for doc in retrieved_docs:
                if doc.metadata.get('has_tables', False) or doc.metadata.get('contains_tables', False):
                    has_tables = True
                    if 'tables' in doc.metadata and isinstance(doc.metadata['tables'], list):
                        table_info.extend(doc.metadata['tables'])

        # Table-specific instructions are added only if tables are present.
        table_instructions = ""
        if has_tables:
            table_instructions = """
TABLE INSTRUCTIONS:
1. The context includes tables marked with [TABLE X] and [/TABLE X] tags.
2. Pay special attention to any numerical or data-related information in these tables.
3. When referencing tables, cite the table number and source.
"""

        # Updated prompt: This prompt now instructs the model to be concise and factual,
        # focusing on direct answers with source citations.
        prompt_text = f"""
You are a precise and factual sports information assistant. 
Your task is to answer the question below using ONLY the provided context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Base your answer ONLY on the provided context.
2. Be concise and direct - provide just the facts that answer the question.
3. Use a clear, structured format with minimal text.
4. Always cite sources with (Source: document name, Page: page number).
5. If the answer cannot be determined from the context, clearly state this.
6. Do not include personal opinions or unnecessary elaboration.
7. {table_instructions}
"""

        # Get response from OpenRouter using DeepSeek Chat v3
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",  # Free high-performance model
            messages=[
                {"role": "system", "content": (
                    "You are a factual sports information assistant for Aspire Academy. "
                    "Provide concise, accurate answers with clear source citations. "
                    "Focus only on the information present in the provided context."
                )},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=1500,   # Shorter responses
            top_p=0.95,
            frequency_penalty=0.0
        )

        # Extract the answer from the API response
        answer = response.choices[0].message.content.strip()

        # Process sources from retrieved_docs (if available)
        sources_text = ""
        if retrieved_docs and len(retrieved_docs) > 0:
            sources = set()
            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'Unknown Source')
                page = doc.metadata.get('page', 'Unknown Page')
                sources.add(f"{source} (Page {page})")
            if sources:
                sources_list = sorted(list(sources))
                sources_text = "**Sources:**\n" + "\n".join(f"- {src}" for src in sources_list)

        return answer, sources_text

    except Exception as e:
        logging.error(f"Error getting answer: {str(e)}")
        return (f"I encountered an error while processing your question: {str(e)}", "")

def get_llm_response(query, context, retrieved_docs=None):
    """
    Wrapper function for get_answer to maintain compatibility with other modules.

    Args:
        query (str): The user's question.
        context (str): The relevant context for answering the question.
        retrieved_docs (list, optional): List of retrieved document objects with metadata.

    Returns:
        str: The answer text only (without sources).
    """
    logging.info(f"get_llm_response called with query: {query[:50]}...")
    answer, _ = get_answer(query, context, retrieved_docs)
    return answer

def get_sql_llm_response(prompt, model="deepseek/deepseek-chat-v3-0324:free", db_type="mysql"):
    """
    Get a response from a language model specifically for SQL query generation.
    Uses DeepSeek Chat v3 model by default for fast response times.

    Args:
        prompt (str): The prompt for SQL generation.
        model (str): The model to use, defaults to DeepSeek Chat v3 which provides quick responses.
        db_type (str): The database type, either "mysql" or "postgres". Defaults to "mysql".

    Returns:
        str: The generated SQL query.
    """
    try:
        # Check for OpenRouter API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logging.error("OPENROUTER_API_KEY not found in environment variables")
            return "ERROR: OpenRouter API key is missing."
        
        # Initialize OpenAI client with OpenRouter base URL
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Set system message based on database type
        if db_type.lower() == "mysql":
            system_message = (
                "You are an expert MySQL query generator. Your task is to convert natural language queries into "
                "precise SQL statements. Pay special attention to field names and values, ensuring exact matches "
                "with database schema. For athletics data, recognize common abbreviations like '100m' as '100 Metres', "
                "'800m' as '800 Metres', etc. Always use LIKE with wildcards for text searches to improve matching. "
                "DO NOT use backticks (`) around table and column names as they can cause issues with some MySQL configurations."
            )
        else:  # postgres
            system_message = (
                "You are an expert PostgreSQL query generator. Your task is to convert natural language queries into "
                "precise SQL statements. Pay special attention to field names and values, ensuring exact matches "
                "with database schema. For athletics data, recognize common abbreviations like '100m' as '100 Metres', "
                "'800m' as '800 Metres', etc. Always use LIKE with wildcards for text searches to improve matching."
            )

        # Get response from OpenRouter using specified model
        response = client.chat.completions.create(
            model=model,  # Use DeepSeek Chat v3 by default for SQL
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Lower temperature for more precise SQL generation
            max_tokens=1000,
            top_p=0.95,
            frequency_penalty=0.0
        )

        # Extract the SQL from the API response
        sql_query = response.choices[0].message.content.strip()
        
        # Clean up the response to extract just the SQL
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
        
        # Remove any markdown code block markers
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Remove backticks only for MySQL queries
        if db_type.lower() == "mysql":
            sql_query = sql_query.replace("`", "")
            logging.info(f"Processed MySQL query (backticks removed): {sql_query}")
        else:
            logging.info(f"Processed PostgreSQL query: {sql_query}")
        
        return sql_query

    except Exception as e:
        logging.error(f"Error getting SQL response: {str(e)}")
        return f"ERROR: {str(e)}"

def get_mysql_llm_response(prompt, model="deepseek/deepseek-chat-v3-0324:free"):
    """
    Wrapper function specifically for MySQL query generation.
    Ensures backticks are removed from the generated SQL.
    
    Args:
        prompt (str): The prompt for SQL generation.
        model (str): The model to use, defaults to DeepSeek Chat v3.
        
    Returns:
        str: The generated MySQL query with backticks removed.
    """
    return get_sql_llm_response(prompt, model, db_type="mysql")

def get_postgres_llm_response(prompt, model="deepseek/deepseek-chat-v3-0324:free"):
    """
    Wrapper function specifically for PostgreSQL query generation.
    Preserves backticks in the generated SQL if present.
    
    Args:
        prompt (str): The prompt for SQL generation.
        model (str): The model to use, defaults to DeepSeek Chat v3.
        
    Returns:
        str: The generated PostgreSQL query.
    """
    return get_sql_llm_response(prompt, model, db_type="postgres")

if __name__ == "__main__":
    # Example usage:
    sample_question = "Can you tell me about the evolution of pole vault competitions and any famous historical records?"
    sample_context = (
        "The dataset includes records from various years. In one entry, a men's pole vault result shows a winning mark of 4.80 meters "
        "at an indoor event in 1990 by Aleksandrs MATUSĒVIČS from Latvia."
    )
    # No retrieved_docs in this simple example; you could pass a list of objects if available.
    response = get_llm_response(sample_question, sample_context)
    print("Generated Response:\n", response)
