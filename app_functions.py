"""
app_functions.py

This module provides utility functions used across the application.
"""

import os
import logging
import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_retrieval_context(docs: List[Document]) -> str:
    """
    Format retrieved documents into a context string for the LLM
    
    Args:
        docs: Collection of retrieved documents
        
    Returns:
        str: Formatted context string
    """
    try:
        context_parts = []
        
        # First add non-parent documents
        for doc in docs:
            if not doc.metadata.get("is_parent", False):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                context_parts.append(f"\nSource: {source}, Page: {page}\nContent: {doc.page_content}")
        
        # Then add parent documents for additional context
        for doc in docs:
            if doc.metadata.get("is_parent", False):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "Unknown")
                child_page = doc.metadata.get("parent_of_page", "Unknown")
                context_parts.append(f"\nParent Context (Source: {source}, Page: {page}, Context for Page: {child_page}):\n{doc.page_content}")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Error formatting context: {str(e)}")
        # Return whatever we can from the docs
        return "\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content'))

def create_prompt_with_context(query: str, context: str) -> str:
    """
    Create a prompt that includes the context and query
    
    Args:
        query: User's question
        context: Retrieved context
        
    Returns:
        str: Complete prompt for LLM
    """
    return f"""
Please answer the following question based on the provided context. 
If the answer cannot be fully determined from the context, say so.

Context:
{context}

Question: {query}

Answer:"""

def extract_and_cite_answer(response: str, context_docs: List[Document]) -> Dict[str, Any]:
    """
    Extract answer and add citations from source documents
    
    Args:
        response: LLM response
        context_docs: Source documents
        
    Returns:
        dict: Answer with citations
    """
    try:
        # Extract sources
        sources = []
        for doc in context_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            sources.append({"source": source, "page": page})
        
        return {
            "answer": response,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error extracting citations: {str(e)}")
        return {
            "answer": response,
            "sources": []
        }

def format_metrics_for_streamlit(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format metrics for display in Streamlit
    
    Args:
        metrics: Raw metrics from retrieval
        
    Returns:
        dict: Formatted metrics for display
    """
    try:
        return {
            "Total Results": metrics.get("total_results", 0),
            "Vector Results": metrics.get("vector_results", 0),
            "SQL Results": metrics.get("sql_results", 0),
            "Table Results": metrics.get("table_results", 0),
            "Parent Documents": metrics.get("parent_count", 0)
        }
    except Exception as e:
        logger.error(f"Error formatting metrics: {str(e)}")
        return {}

def save_dataframe(df: pd.DataFrame, query: str, query_type: str = "nl_query", reuse_current: bool = False) -> Optional[str]:
    """
    Save a dataframe to a parquet file in the temp_data directory
    
    Args:
        df: DataFrame to save
        query: Query that generated the data
        query_type: Type of query (nl_query or mysql_nl_query)
        reuse_current: Whether to reuse the current file path from session state
        
    Returns:
        str: Path to saved file, or None if save failed
    """
    try:
        import streamlit as st
        
        # Create temp_data directory if it doesn't exist
        temp_dir = Path("temp_data")
        temp_dir.mkdir(exist_ok=True)
        
        # If reusing current file path
        if reuse_current and hasattr(st.session_state, 'current_temp_file') and st.session_state.current_temp_file:
            filepath = st.session_state.current_temp_file
            logger.info(f"Reusing existing file path: {filepath}")
        else:
            # Create unique filename with timestamp and random suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_suffix = os.urandom(3).hex()
            filename = f"{query_type}_{timestamp}_{random_suffix}.parquet"
            filepath = temp_dir / filename
            
            # Update session state with new file path
            if hasattr(st, 'session_state'):
                st.session_state.current_temp_file = str(filepath)
        
        # Save dataframe
        df.to_parquet(str(filepath))
        logger.info(f"Saved dataframe to {filepath}")
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Error saving dataframe: {str(e)}")
        return None

def get_temp_files() -> List[str]:
    """
    Get list of temporary data files
    
    Returns:
        list: List of filenames in temp_data directory
    """
    try:
        temp_dir = Path("temp_data")
        if not temp_dir.exists():
            return []
            
        files = [f.name for f in temp_dir.glob("*.parquet")]
        return sorted(files, reverse=True)
        
    except Exception as e:
        logger.error(f"Error getting temp files: {str(e)}")
        return []

def clean_temp_files(keep_current: bool = True) -> None:
    """
    Clean up temporary data files
    
    Args:
        keep_current: Whether to keep the current file
    """
    try:
        temp_dir = Path("temp_data")
        if not temp_dir.exists():
            return
            
        files = list(temp_dir.glob("*.parquet"))
        
        # Sort by modification time
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep most recent file if requested
        if keep_current and files:
            files = files[1:]
            
        # Delete files
        for file in files:
            try:
                file.unlink()
                logger.info(f"Deleted temp file: {file}")
            except Exception as e:
                logger.error(f"Error deleting {file}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error cleaning temp files: {str(e)}")

def get_answer_from_documents(query: str, docs: List[Document], llm) -> str:
    """
    Get an answer from documents using the provided LLM
    
    Args:
        query: User's question
        docs: List of relevant documents
        llm: Language model to use
        
    Returns:
        str: Generated answer
    """
    try:
        # Format context from documents
        context = format_retrieval_context(docs)
        
        # Create prompt
        prompt = create_prompt_with_context(query, context)
        
        # Get answer from LLM
        answer = llm(prompt)
        
        return answer
        
    except Exception as e:
        logger.error(f"Error getting answer from documents: {str(e)}")
        return "Sorry, I was unable to generate an answer from the documents."

def analyze_data(df, question, sample_rows=None):
    """
    Analyze data using the selected backend in session state
    
    Args:
        df: DataFrame to analyze
        question: Question to answer
        sample_rows: Number of rows to sample (None for all)
        
    Returns:
        dict: Analysis results with text and plots
    """
    import streamlit as st
    
    # Default to OpenRouter if not specified
    backend = st.session_state.get('analysis_backend', 'openrouter')
    
    if backend == 'pandasai':
        return analyze_data_with_pandasai(df, question, sample_rows)
    elif backend == 'hybrid':
        return analyze_data_hybrid(df, question, sample_rows)
    else:  # Default to OpenRouter
        return analyze_data_with_openrouter(df, question, sample_rows)

def analyze_data_with_openrouter(df, question, sample_rows=None):
    """
    Analyze database data using LangChain's standard components.
    
    This function uses LangChain's PromptTemplate and LLMChain to analyze the provided dataframe.
    It converts the dataframe to a text representation and passes it to the LLM along with
    strict prompt instructions to minimize hallucination.
    
    Args:
        df: DataFrame to analyze
        question: Question to answer about the data
        sample_rows: Number of rows to include in the sample (None for all, but limited to 50 max)
        
    Returns:
        dict: Analysis results with text and plots
    """
    import os
    import uuid
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import io
    
    try:
        # Create plots directory if it doesn't exist
        plots_dir = Path("temp_data/plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Convert to pandas if it's a polars DataFrame
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
            
        # Log dataset info
        logger.info(f"Analyzing dataset with shape: {df.shape}")
        if 'Event' in df.columns:
            events = df['Event'].unique().tolist()
            logger.info(f"Events in dataset: {events}")
            
        # Get OpenRouter API key
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            return {"text": "OpenRouter API key not found. Please add it to your .env file as OPENROUTER_API_KEY.", "plots": []}
        
        # Log API call start time
        start_time = time.time()
        logger.info(f"Starting LangChain analysis at {start_time}")
        
        # Import LangChain components
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain.chat_models import ChatOpenAI
        
        # Create a custom LLM that uses OpenRouter
        llm = ChatOpenAI(
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model_name="deepseek/deepseek-chat-v3-0324:free",
            temperature=0.2
        )
        
        # Create events list string
        events_list_str = "No events found in dataset"
        if 'Event' in df.columns:
            events = df['Event'].unique().tolist()
            events_list_str = ", ".join([f"'{event}'" for event in events])
        
        # Use the full dataset in the prompt as requested by the user
        # This will provide the LLM with complete information
        # Note: For extremely large datasets, this might cause token overflow
        
        # Convert the entire dataframe to markdown table
        df_text = df.to_markdown(index=False)
        
        # Get dataframe statistics
        stats_buffer = io.StringIO()
        df.describe().to_markdown(stats_buffer)
        stats_text = stats_buffer.getvalue()
        
        # Create the prompt template
        template = """
        You are a knowledgeable sports data analyst with expertise in athletics. 
        
        I have a dataset with the following information:
        - Shape: {rows} rows x {cols} columns
        - Columns: {columns}
        - This dataset ONLY contains the following events: {events}
        
        Here is the COMPLETE dataset ({rows} rows):
        {df_sample}
        
        Here are the statistics of the numerical columns:
        {stats}
        
        Question: {question}
        
        CRITICAL INSTRUCTIONS:
        1. ONLY analyze the data provided in this dataset. Do not make assumptions about data not present.
        2. This dataset ONLY contains the following events: {events}. Do NOT discuss any other events.
        3. If asked for rankings or top results, ALWAYS include specific names and values.
        4. For time-based results, convert text formats to seconds for proper comparison.
        5. Keep your tone friendly but factual.
        6. Make sure your analysis is thorough and considers the ENTIRE dataset of {rows} rows, not just the sample shown above.
        7. When discussing records or statistics, refer mainly to what's in this specific dataset.
        8. ALWAYS include a concise summary at the end of your analysis that highlights the key findings.
        
        Provide a detailed analysis answering the question.
        """
        
        # Create the prompt
        prompt = PromptTemplate(
            input_variables=["rows", "cols", "columns", "events", "df_sample", "stats", "question"],
            template=template
        )
        
        # Create the chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Run the chain
        result = chain.run(
            rows=df.shape[0],
            cols=df.shape[1],
            columns=", ".join(df.columns),
            events=events_list_str,
            df_sample=df_text,
            stats=stats_text,
            question=question
        )
        
        # Log API call duration
        end_time = time.time()
        logger.info(f"LangChain analysis completed in {end_time - start_time:.2f} seconds")
        
        # Generate a simple plot if possible and appropriate
        plots = []
        try:
            # Only create plots for certain types of questions
            plot_keywords = ["top", "best", "fastest", "slowest", "highest", "lowest", "compare", "trend", "distribution"]
            should_plot = any(keyword in question.lower() for keyword in plot_keywords)
            
            if should_plot and 'Event' in df.columns:
                # Create a simple plot based on the question
                plt.figure(figsize=(10, 6))
                
                if 'Result' in df.columns:
                    # For time-based results, try to convert to numeric
                    try:
                        # Function to convert time strings to seconds
                        def to_seconds(time_str):
                            if pd.isna(time_str):
                                return np.nan
                            if isinstance(time_str, (int, float)):
                                return float(time_str)
                            if ':' in str(time_str):
                                parts = str(time_str).split(':')
                                if len(parts) == 2:
                                    return float(parts[0]) * 60 + float(parts[1])
                                elif len(parts) == 3:
                                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                            return float(time_str)
                        
                        # Apply conversion
                        df['Result_Seconds'] = df['Result'].apply(to_seconds)
                        
                        # Sort by result
                        df_sorted = df.sort_values('Result_Seconds').head(10)
                        
                        # Plot
                        plt.barh(df_sorted['Athlete_Name'], df_sorted['Result_Seconds'])
                        plt.xlabel('Time (seconds)')
                        plt.ylabel('Athlete')
                        plt.title('Top 10 Fastest Times')
                        plt.tight_layout()
                        
                        # Save the plot
                        plot_path = plots_dir / f"plot_{uuid.uuid4().hex}.png"
                        plt.savefig(plot_path, bbox_inches='tight')
                        plt.close()
                        plots.append(str(plot_path))
                    except Exception as plot_error:
                        logger.error(f"Error creating time plot: {str(plot_error)}")
                
                # Try a count plot of events if multiple events
                if len(events) > 1:
                    try:
                        event_counts = df['Event'].value_counts()
                        plt.figure(figsize=(10, 6))
                        plt.bar(event_counts.index, event_counts.values)
                        plt.xlabel('Event')
                        plt.ylabel('Count')
                        plt.title('Event Distribution')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        # Save the plot
                        plot_path = plots_dir / f"plot_{uuid.uuid4().hex}.png"
                        plt.savefig(plot_path, bbox_inches='tight')
                        plt.close()
                        plots.append(str(plot_path))
                    except Exception as plot_error:
                        logger.error(f"Error creating event count plot: {str(plot_error)}")
        except Exception as plot_error:
            logger.error(f"Error creating plots: {str(plot_error)}")
        
        return {"text": result, "plots": plots}
            
    except Exception as e:
        logger.error(f"Error in data analysis: {str(e)}")
        return {"text": f"Error analyzing data: {str(e)}", "plots": []}

def analyze_data_with_pandasai(df, question, sample_rows=None):
    """
    Analyze data using PandasAI
    
    Args:
        df: DataFrame to analyze
        question: Question to answer
        sample_rows: Number of rows to sample (None for all)
        
    Returns:
        dict: Analysis results with text and plots
    """
    import os
    from pandasai import SmartDataframe
    from pandasai.llm.openai import OpenAI as PandasAIOpenAI
    from pathlib import Path
    
    try:
        # Create plots directory if it doesn't exist
        plots_dir = Path("temp_data/plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Always use the full dataset regardless of sample_rows parameter
        sample_df = df
        
        # Convert to pandas if it's a polars DataFrame
        if hasattr(sample_df, 'to_pandas'):
            sample_df = sample_df.to_pandas()
        
        # Get OpenRouter API key
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            return {"text": "OpenRouter API key not found. Please add it to your .env file as OPENROUTER_API_KEY.", "plots": []}
        
        # Initialize PandasAI
        llm = PandasAIOpenAI(
            api_token=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            model="deepseek/deepseek-chat-v3-0324:free",
            custom_headers={"HTTP-Referer": "https://mcp-learning.example.com"}
        )
        
        # Create SmartDataframe
        smart_df = SmartDataframe(sample_df, config={"llm": llm, "save_charts": True, "save_charts_path": str(plots_dir)})
        
        # Run analysis
        result = smart_df.chat(question)
        
        # Get plots
        plots = []
        for file in plots_dir.glob("*.png"):
            if file.stat().st_mtime > (time.time() - 60):  # Only include plots created in the last minute
                plots.append(str(file))
        
        return {"text": str(result), "plots": plots}
    except Exception as e:
        logger.error(f"Error analyzing data with PandasAI: {str(e)}")
        return {"text": f"Error analyzing data: {str(e)}", "plots": []}

def analyze_data_hybrid(df, question, sample_rows=None):
    """
    Analyze data using both PandasAI and OpenRouter, then combine results
    
    Args:
        df: DataFrame to analyze
        question: Question to answer
        sample_rows: Number of rows to sample (None for all)
        
    Returns:
        dict: Analysis results with text and plots
    """
    import streamlit as st
    import time
    
    try:
        # Run both analyses
        pandasai_result = analyze_data_with_pandasai(df, question, sample_rows)
        openrouter_result = analyze_data_with_openrouter(df, question, sample_rows)
        
        # Combine plots
        plots = pandasai_result["plots"] + openrouter_result["plots"]
        
        # Combine text
        combined_text = f"""
        # Combined Analysis

        ## Analysis from PandasAI:
        {pandasai_result["text"]}

        ## Analysis from OpenRouter:
        {openrouter_result["text"]}
        """
        
        # Store original analyses for reference
        timestamp = time.time()
        if not hasattr(st.session_state, 'hybrid_original_analyses'):
            st.session_state.hybrid_original_analyses = {}
        st.session_state.hybrid_original_analyses[timestamp] = {
            "pandasai": pandasai_result["text"],
            "openrouter": openrouter_result["text"]
        }
        st.session_state.latest_hybrid_analysis = timestamp
        
        return {"text": combined_text, "plots": plots}
    except Exception as e:
        logger.error(f"Error in hybrid analysis: {str(e)}")
        return {"text": f"Error in hybrid analysis: {str(e)}", "plots": []}
