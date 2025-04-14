"""
app_ranking.py

This module handles document ranking and relevance scoring for the RAG system.
It provides algorithms to determine the most relevant documents for a given query
and detect query types for specialized handling.

Key functions:
- rank_docs_by_relevance: Ranks documents by keyword relevance using a multi-factor
  scoring algorithm that considers exact matches, word boundaries, position in text,
  and multiple keyword presence.
- is_table_oriented_query: Detects if a query is likely asking for tabular/structured
  data by analyzing keywords, patterns, and numerical indicators.
"""

import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rank_docs_by_relevance(docs, query):
    """
    Rank documents by keyword relevance to the query
    
    Args:
        docs: List of Document objects
        query: User query string
        
    Returns:
        list: Sorted list of Document objects by relevance
    """
    # If no docs or query, return as is
    if not docs or not query:
        return docs
        
    # Extract keywords from query, lowercase
    keywords = [k.lower() for k in query.split() if len(k) > 3]
    
    # If no meaningful keywords, return docs as is
    if not keywords:
        return docs
    
    # Calculate relevance scores for each document
    scored_docs = []
    for doc in docs:
        try:
            score = 0
            # Skip if content is empty
            if not doc.page_content:
                continue
                
            content = doc.page_content.lower()
            
            # Score based on keyword presence
            for keyword in keywords:
                if keyword in content:
                    # Count occurrences with word boundaries for more precision
                    # This helps prioritize exact matches over partial matches
                    try:
                        count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', content))
                        score += count * 2  # Give higher weight to exact keyword matches
                    except:
                        # Fallback if regex fails
                        score += content.count(keyword) * 2
                    
                    # If keyword appears in first sentence, give bonus points (likely more relevant)
                    try:
                        first_sentence = content.split('.')[0] if '.' in content else content
                        if keyword in first_sentence:
                            score += 5
                    except:
                        # Skip this bonus if we can't split sentences
                        pass
            
            # Give bonus for having multiple keywords
            try:
                keyword_matches = sum(1 for keyword in keywords if keyword in content)
                if keyword_matches > 1:
                    score += keyword_matches * 3
            except:
                # Skip this bonus if counting fails
                pass
                
            # Score exact phrase matches even higher
            if query.lower() in content:
                score += 10
                
            # Prioritize documents with relevant headings/beginnings
            try:
                if any(keyword in content[:100].lower() for keyword in keywords):
                    score += 8
            except:
                # Skip this bonus if slicing fails
                pass
                
            # Add document with its score
            scored_docs.append((doc, score))
        except Exception as e:
            logger.error(f"Error scoring document: {str(e)}")
            # Add with zero score so it at least appears in results
            scored_docs.append((doc, 0))
    
    try:
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.error(f"Error sorting documents by score: {str(e)}")
        # Return unsorted if sorting fails
        return [doc for doc, _ in scored_docs]
    
    # Return sorted documents
    return [doc for doc, score in scored_docs]

def is_table_oriented_query(query):
    """
    Analyze if a query is likely asking for tabular/structured data
    
    Args:
        query (str): The user's query
        
    Returns:
        bool: True if likely table-oriented, False otherwise
    """
    # List of keywords that suggest the query is looking for structured/tabular data
    table_keywords = [
        'table', 'tabular', 'row', 'column', 'cell',
        'spreadsheet', 'excel', 'csv', 'tsv', 
        'data', 'dataset', 'statistics', 'stats',
        'numbers', 'figures', 'metrics', 'measurements',
        'chart', 'graph', 'plot', 'diagram',
        'compare', 'comparison', 'difference', 'similarities',
        'highest', 'lowest', 'maximum', 'minimum', 'average', 'median',
        'percentage', 'ratio', 'proportion', 'distribution',
        'ranking', 'ranked', 'rank', 'list',
        'how many', 'what percentage', 'values', 'numeric'
    ]
    
    # Numerical pattern indicators
    numeric_patterns = [
        r'\d+(\.\d+)?%',  # Percentage pattern
        r'\$\d+(\.\d+)?',  # Money pattern
        r'\d{4}-\d{2}-\d{2}',  # Date pattern
        r'\d+x\d+',  # Dimension pattern
    ]
    
    # Check for table keywords
    query_lower = query.lower()
    for keyword in table_keywords:
        if keyword in query_lower:
            return True
            
    # Check for numeric patterns
    for pattern in numeric_patterns:
        if re.search(pattern, query):
            return True
    
    # Check for comparison words with numbers
    comparison_with_numbers = re.search(r'(compare|difference|between|vs|versus).*\d+', query_lower)
    if comparison_with_numbers:
        return True
        
    return False
