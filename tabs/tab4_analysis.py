"""
Data Analysis tab for MCP Dashboard.
Provides AI-powered analysis of dataframes with multiple backend options.
"""

import os
import streamlit as st
import pandas as pd
import polars as pl
from PIL import Image
import traceback
from datetime import datetime
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI as PandasAIOpenAI
import openai
from openai import OpenAI

from app_functions import (
    analyze_data,
    analyze_data_with_openrouter,
    analyze_data_with_pandasai,
    analyze_data_hybrid,
    save_dataframe,
    get_temp_files,
    clean_temp_files
)

# Path to temp directory
temp_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp_data')
os.makedirs(temp_data_dir, exist_ok=True)

def show_analysis_tab():
    """
    Display the Data Analysis tab that allows users to analyze query results with AI
    """
    st.header("Data Analysis")
    
    # Create three columns for a clearer info panel at the top
    info_col1, info_col2, info_col3 = st.columns([1,1,1])
    with info_col1:
        st.info("📊 Select data sources →")
    with info_col2:
        st.info("❓ Ask questions about your data →")
    with info_col3:
        st.info("📈 Get AI-powered analysis →")
    
    # Add tabs to better organize the different ways to select/analyze data
    data_tab, saved_tab, multi_tab = st.tabs(["Current Data", "Saved Files", "Multi-DataFrame Analysis"])
    
    with data_tab:
        if hasattr(st.session_state, 'last_query_df') and isinstance(st.session_state.last_query_df, (pd.DataFrame, pl.DataFrame)):
            df = st.session_state.last_query_df
            if st.session_state.use_polars and isinstance(df, pd.DataFrame):
                df = pl.from_pandas(df)
                st.session_state.last_query_df = df
            elif not st.session_state.use_polars and isinstance(df, pl.DataFrame):
                df = df.to_pandas()
                st.session_state.last_query_df = df
            
            st.success(f"✅ **Active Dataset:** {len(df)} rows × {len(df.columns)} columns")
            
            if st.session_state.use_polars:
                st.dataframe(df.head(5).to_pandas(), use_container_width=True)
            else:
                st.dataframe(df.head(5), use_container_width=True)
            
            with st.form("analysis_form"):
                st.subheader("Ask Questions About Your Data")
                data_question = st.text_area("What would you like to know about this data?", 
                                          placeholder="e.g., What are the main patterns in this data? or Summarize the key statistics.")
                
                col1, col2 = st.columns([1,4])
                with col1:
                    analyze_button = st.form_submit_button("🔍 Analyze Data", type="primary", use_container_width=True)
                with col2:
                    if st.session_state.analysis_backend in ["openrouter", "hybrid"]:
                        use_full_dataset = st.checkbox("Analyze full dataset (may be slower but more accurate)", value=True)
                    else:
                        use_full_dataset = True
            
            if analyze_button and data_question:
                with st.spinner("Analyzing data with AI..."):
                    sample_size = None if use_full_dataset else 5
                    analysis = analyze_data(df, data_question, sample_rows=sample_size)
                    st.markdown("### Analysis")
                    st.markdown(analysis["text"])
                    if analysis["plots"] and len(analysis["plots"]) > 0:
                        st.markdown("### Visualizations")
                        plot_cols = st.columns(min(3, len(analysis["plots"])))
                        for i, plot_path in enumerate(analysis["plots"]):
                            try:
                                col_idx = i % len(plot_cols)
                                with plot_cols[col_idx]:
                                    img = Image.open(plot_path)
                                    st.image(img, caption=f"Plot {i+1}", use_column_width=True)
                            except Exception as e:
                                st.error(f"Error displaying plot {i+1}: {str(e)}")
                    
                    if st.session_state.current_temp_file is None:
                        filepath = save_dataframe(df, "analysis", reuse_current=False)
                    st.session_state.last_analysis = analysis["text"]
                    st.session_state.last_analysis_plots = analysis["plots"]
                    st.session_state.show_followup = True
                    
                    if st.session_state.analysis_backend == "hybrid" and hasattr(st.session_state, 'latest_hybrid_analysis'):
                        timestamp = st.session_state.latest_hybrid_analysis
                        if timestamp in st.session_state.hybrid_original_analyses:
                            with st.expander("View Individual Analyses"):
                                individual_analyses = st.session_state.hybrid_original_analyses[timestamp]
                                st.markdown("### PandasAI Analysis")
                                st.markdown(individual_analyses["pandasai"])
                                st.markdown("### OpenRouter Analysis")
                                st.markdown(individual_analyses["openrouter"])
            
            if hasattr(st.session_state, 'show_followup') and st.session_state.show_followup:
                with st.form("followup_form"):
                    st.subheader("Follow-up Question")
                    followup_question = st.text_area("Ask a follow-up question based on the analysis:", 
                                                placeholder="e.g., Can you provide more details about...?")
                    followup_button = st.form_submit_button("Get Follow-up Analysis", type="primary")
                
                if followup_button and followup_question:
                    with st.spinner("Analyzing further..."):
                        combined_question = f"Previous analysis: {st.session_state.last_analysis}\n\nFollow-up question: {followup_question}"
                        sample_size = None if use_full_dataset else 5
                        if st.session_state.current_temp_file:
                            save_dataframe(df, "analysis", reuse_current=True)
                        followup_analysis = analyze_data(df, combined_question, sample_rows=sample_size)
                        st.markdown("### Follow-up Analysis")
                        st.markdown(followup_analysis["text"])
                        if followup_analysis["plots"] and len(followup_analysis["plots"]) > 0:
                            st.markdown("### Follow-up Visualizations")
                            plot_cols = st.columns(min(3, len(followup_analysis["plots"])))
                            for i, plot_path in enumerate(followup_analysis["plots"]):
                                try:
                                    col_idx = i % len(plot_cols)
                                    with plot_cols[col_idx]:
                                        img = Image.open(plot_path)
                                        st.image(img, caption=f"Plot {i+1}", use_column_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying plot {i+1}: {str(e)}")
            
            st.divider()
            if st.button("🗑️ Clear Current Data"):
                if hasattr(st.session_state, 'last_query_df'):
                    del st.session_state.last_query_df
                if hasattr(st.session_state, 'last_query_filepath'):
                    del st.session_state.last_query_filepath
                if hasattr(st.session_state, 'last_analysis'):
                    del st.session_state.last_analysis
                if hasattr(st.session_state, 'show_followup'):
                    del st.session_state.show_followup
                st.session_state.current_temp_file = None
                clean_temp_files(keep_current=False)
                st.rerun()
        else:
            st.warning("⚠️ No data currently loaded. Please run a query in the PostgreSQL or MySQL Explorer tab and save the results for analysis, or load a saved file below.")
    
    with saved_tab:
        st.subheader("Previously Saved Data Files")
        if os.path.exists(temp_data_dir) and os.listdir(temp_data_dir):
            csv_files = [f for f in os.listdir(temp_data_dir) if f.endswith('.csv')]
            parquet_files = [f for f in os.listdir(temp_data_dir) if f.endswith('.parquet')]
            if csv_files or parquet_files:
                col1, col2 = st.columns([1, 3])
                with col1:
                    file_type = st.radio("File type:", ["CSV", "Parquet"], 
                                      index=1 if st.session_state.use_polars else 0,
                                      help="CSV files are created with pandas, Parquet files with Polars")
                with col2:
                    selected_file = None
                    if file_type == "CSV" and csv_files:
                        selected_file = st.selectbox("Select a saved file to load:", csv_files)
                    elif file_type == "Parquet" and parquet_files:
                        selected_file = st.selectbox("Select a saved file to load:", parquet_files)
                
                if selected_file and st.button("📂 Load Selected File", type="primary"):
                    try:
                        file_path = os.path.join(temp_data_dir, selected_file)
                        if selected_file.endswith('.csv'):
                            if st.session_state.use_polars:
                                df = pl.read_csv(file_path)
                            else:
                                df = pd.read_csv(file_path)
                        else:
                            if st.session_state.use_polars:
                                df = pl.read_parquet(file_path)
                            else:
                                df = pd.read_parquet(file_path)
                        st.session_state.last_query_df = df
                        st.session_state.last_query_filepath = file_path
                        st.session_state.current_temp_file = file_path
                        st.success(f"✅ Loaded {selected_file} with {len(df)} rows and {len(df.columns)} columns")
                        st.info("Switch to the 'Current Data' tab to analyze this dataset")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
            else:
                st.info("No saved data files found.")
    
    with multi_tab:
        st.subheader("Multi-DataFrame Analysis")
        st.markdown("Load multiple CSV or Parquet files for combined analysis")
        
        if os.path.exists(temp_data_dir) and os.listdir(temp_data_dir):
            data_files = [f for f in os.listdir(temp_data_dir) if f.endswith(('.csv', '.parquet'))]
            if data_files:
                selected_files = st.multiselect(
                    "Select files to analyze together:",
                    data_files,
                    help="Select multiple files to analyze relationships between them"
                )
                if selected_files and len(selected_files) > 1:
                    if st.button("📊 Load Selected Files for Multi-Analysis", type="primary"):
                        try:
                            dfs = []
                            for file in selected_files:
                                file_path = os.path.join(temp_data_dir, file)
                                if file.endswith('.csv'):
                                    if st.session_state.use_polars:
                                        df = pl.read_csv(file_path)
                                    else:
                                        df = pd.read_csv(file_path)
                                else:
                                    if st.session_state.use_polars:
                                        df = pl.read_parquet(file_path)
                                    else:
                                        df = pd.read_parquet(file_path)
                                file_id = os.path.splitext(file)[0]
                                dfs.append((file_id, df))
                            st.session_state.multi_dfs = dfs
                            st.success(f"✅ Loaded {len(dfs)} dataframes for multi-analysis")
                            st.subheader("Loaded DataFrames")
                            for name, df in dfs:
                                st.write(f"**{name}**: {len(df)} rows, {len(df.columns) if hasattr(df, 'columns') else 0} columns")
                            
                            with st.form("multi_analysis_form"):
                                st.subheader("Ask Questions About Multiple DataFrames")
                                multi_question = st.text_area(
                                    "What would you like to know about these datasets?",
                                    placeholder="e.g., What's the relationship between these datasets? or Compare the key metrics across these tables."
                                )
                                multi_analyze_button = st.form_submit_button("🔍 Analyze Multiple DataFrames", type="primary")
                            
                            if multi_analyze_button and multi_question:
                                with st.spinner("Analyzing relationships between dataframes..."):
                                    llm = OpenAI(
                                        api_token=os.getenv('OPENROUTER_API_KEY'),
                                        base_url="https://openrouter.ai/api/v1",
                                        model="deepseek/deepseek-chat-v3-0324:free",
                                        custom_headers={"HTTP-Referer": "https://mcp-learning.example.com"}
                                    )
                                    smart_dfs = []
                                    for name, df in dfs:
                                        if st.session_state.use_polars:
                                            df = df.to_pandas()
                                        smart_dfs.append(SmartDataframe(df, name=name, config={"llm": llm}))
                                    primary_df = smart_dfs[0]
                                    context = f"I have {len(smart_dfs)} related dataframes with the following names and structures:\n\n"
                                    for i, smart_df in enumerate(smart_dfs):
                                        df_name = smart_df._name
                                        df = smart_df._df
                                        context += f"DataFrame {i+1}: {df_name}\n"
                                        context += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                                        context += f"Columns: {', '.join(df.columns.tolist())}\n\n"
                                    enhanced_question = f"{context}\n\nQuestion: {multi_question}"
                                    result = primary_df.chat(enhanced_question)
                                    st.markdown("### Analysis Results")
                                    st.markdown(str(result))
                        except Exception as e:
                            st.error(f"Error loading files for multi-analysis: {str(e)}")
                            st.error(traceback.format_exc())
            else:
                st.info("No saved data files found.")
        else:
            st.info("No data directory found.")
            
    with st.sidebar:
        # Add a help section at the bottom of the sidebar
        st.divider()
        with st.expander("🔍 How to use Data Analysis", expanded=False):
            st.markdown("""
            ### Quick Guide
            
            1. **Load Data** - Run a query in PostgreSQL/MySQL tab and save for analysis, or load a saved file
            2. **Ask Questions** - Type your question about the data in the text box
            3. **Choose Analysis Method** - Use the sidebar to select your preferred analysis backend
            4. **View Results** - See insights and any generated visualizations
            
            ### Tips
            - Polars is faster for large datasets
            - OpenRouter gives reliable text analysis
            - PandasAI is best for generating visualizations
            - Hybrid combines both approaches
            """) 