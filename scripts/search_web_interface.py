#!/usr/bin/env python3
"""
Simple web interface for the Fashion Retail Search Engine
Run this after starting the search engine server
"""

import streamlit as st
import requests
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def check_server_status():
    """Check if the search engine server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def search_products(query: str, top_k: int = 3):
    """Search for products using the API"""
    try:
        response = requests.get(
            f"{BASE_URL}/search", 
            params={"query": query, "top_k": top_k},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Search failed: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None

def get_recommendations(product_id: str):
    """Get product recommendations"""
    try:
        response = requests.get(f"{BASE_URL}/recommend/{product_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Recommendations failed: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None

def load_evaluation_results():
    """Load SBERT evaluation results if available"""
    eval_file = Path("/home/artisans15/projects/fashion_retail_analytics/evaluation_results/sbert_evaluation_results.json")
    if eval_file.exists():
        try:
            with open(eval_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load evaluation results: {e}")
    return None

def display_evaluation_metrics():
    """Display SBERT model evaluation metrics"""
    eval_results = load_evaluation_results()
    
    if eval_results is None:
        st.info("🔄 Evaluation results not found. Run the evaluation script to generate metrics.")
        st.code("python scripts/evaluate_sbert.py", language="bash")
        return
    
    st.header("🎯 SBERT Model Performance Metrics")
    st.markdown("These metrics show how well our search engine performs on test queries.")
    
    agg = eval_results['aggregate_metrics']
    
    # Key Performance Indicators
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Precision@3", 
            f"{agg['avg_precision@3']:.3f}",
            help="Proportion of top-3 results that are actually relevant"
        )
    
    with col2:
        st.metric(
            "Recall@3", 
            f"{agg['avg_recall@3']:.3f}",
            help="Proportion of relevant items found in top-3 results"
        )
    
    with col3:
        st.metric(
            "F1-Score@3", 
            f"{agg['avg_f1@3']:.3f}",
            help="Harmonic mean of precision and recall"
        )
    
    with col4:
        st.metric(
            "Avg Response Time", 
            f"{agg['avg_mean_latency_ms']:.1f}ms",
            help="Average time to process a search query"
        )
    
    # Performance by K values
    st.subheader("📈 Precision & Recall by K")
    
    k_values = [1, 3, 5, 10]
    metrics_data = []
    
    for k in k_values:
        metrics_data.append({
            'K': k,
            'Precision': agg[f'avg_precision@{k}'],
            'Recall': agg[f'avg_recall@{k}'],
            'F1-Score': agg[f'avg_f1@{k}']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.line_chart(metrics_df.set_index('K')[['Precision', 'Recall']])
    
    with col2:
        st.bar_chart(metrics_df.set_index('K')['F1-Score'])
    
    # Category Performance
    if 'category_metrics' in eval_results:
        st.subheader("🏷️ Performance by Category")
        
        cat_data = []
        for category, metrics in eval_results['category_metrics'].items():
            cat_data.append({
                'Category': category,
                'Precision@3': metrics['avg_precision@3'],
                'Recall@3': metrics['avg_recall@3'],
                'F1@3': metrics['avg_f1@3'],
                'Latency (ms)': metrics['avg_mean_latency_ms']
            })
        
        cat_df = pd.DataFrame(cat_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(cat_df.set_index('Category')['Precision@3'])
            st.caption("Precision@3 by Category")
        
        with col2:
            st.bar_chart(cat_df.set_index('Category')['Latency (ms)'])
            st.caption("Response Time by Category")
        
        # Detailed table
        st.subheader("📋 Detailed Performance Table")
        st.dataframe(cat_df, use_container_width=True)
    
    # Model Information
    if 'dataset_info' in eval_results:
        st.subheader("🤖 Model Information")
        dataset_info = eval_results['dataset_info']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Model:** {dataset_info.get('model_name', 'SBERT')}")
            st.info(f"**Dataset Size:** {dataset_info['total_products']} products")
        
        with col2:
            st.info(f"**Test Queries:** {len(eval_results['individual_results'])}")
            st.info(f"**Categories:** {len(dataset_info['categories'])}")
    
    # Performance Rating
    st.subheader("⭐ Overall Performance Rating")
    
    # Calculate overall score
    overall_score = (agg['avg_precision@3'] + agg['avg_recall@3'] + agg['avg_f1@3']) / 3
    latency_score = min(1.0, 50 / agg['avg_mean_latency_ms'])  # Better if latency < 50ms
    
    combined_score = (overall_score * 0.8 + latency_score * 0.2)
    
    if combined_score >= 0.8:
        rating = "🌟 Excellent"
        color = "green"
    elif combined_score >= 0.6:
        rating = "⭐ Good"
        color = "blue"
    elif combined_score >= 0.4:
        rating = "⚡ Fair"
        color = "orange"
    else:
        rating = "🔧 Needs Improvement"
        color = "red"
    
    st.markdown(f"**Overall Rating:** :{color}[{rating}]")
    st.progress(combined_score)
    
    # Show evaluation image if available
    eval_chart = Path("/home/artisans15/projects/fashion_retail_analytics/evaluation_results/sbert_evaluation_charts.png")
    if eval_chart.exists():
        st.subheader("📊 Evaluation Charts")
        st.image(str(eval_chart), caption="SBERT Model Evaluation Results", use_column_width=True)

def display_product_card(product: Dict[str, Any], show_score: bool = True):
    """Display a product in a nice card format"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Placeholder for product image (you can add actual images later)
        st.markdown("🛍️")
    
    with col2:
        st.markdown(f"**{product['product_name']}**")
        st.markdown(f"*{product['brand']}*")
        
        cols = st.columns(3)
        with cols[0]:
            st.markdown(f"**Category:** {product['product_category']}")
        with cols[1]:
            st.markdown(f"**Type:** {product['product_type']}")
        with cols[2]:
            st.markdown(f"**Price:** ₹{product['price_final']}")
        
        if show_score and 'final_score' in product:
            st.markdown(f"**Relevance Score:** {product['final_score']:.4f}")
        
        st.markdown("---")

def main():
    st.set_page_config(
        page_title="Fashion Retail Search Engine",
        page_icon="🛍️",
        layout="wide"
    )
    
    st.title("🛍️ Fashion Retail Search Engine")
    st.markdown("Search for fashion products and get the top 3 closest recommendations!")
    
    # Check server status
    if not check_server_status():
        st.error("❌ Search engine server is not running!")
        st.info("Please start the server first by running: `python scripts/search.py`")
        return
    
    st.success("✅ Search engine server is running!")
    
    # Sidebar
    st.sidebar.header("Search Options")
    top_k = st.sidebar.slider("Number of results", 1, 10, 3)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🔍 Search Products", "🎯 Model Performance", "📊 Dataset Statistics"])
    
    # Tab 1: Search Interface
    with tab1:
        # Search input
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter your search query:", placeholder="e.g., party heels, work laptop bag, kids winter jackets")
        with col2:
            search_button = st.button("Search", type="primary")
        
        # Example queries
        st.markdown("**Example queries:** party heels, work laptop bag, kids winter jackets, navy running shoes, cotton t-shirt")
        
        if search_button and query:
            with st.spinner("Searching..."):
                results = search_products(query, top_k)
                
                if results and results['results']:
                    # Show search performance
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.success(f"✅ Found {results['total_results']} results")
                    with col2:
                        latency = results['search_time_ms']
                        if latency < 50:
                            st.success(f"⚡ {latency}ms (Excellent)")
                        elif latency < 100:
                            st.info(f"🔋 {latency}ms (Good)")
                        else:
                            st.warning(f"🐌 {latency}ms (Slow)")
                    with col3:
                        # Load evaluation metrics for comparison
                        eval_results = load_evaluation_results()
                        if eval_results:
                            avg_precision = eval_results['aggregate_metrics']['avg_precision@3']
                            st.info(f"🎯 Avg Precision@3: {avg_precision:.3f}")
                        else:
                            st.info("🎯 Model metrics available in Performance tab")
                    
                    st.markdown("---")
                    
                    # Display results
                    for i, product in enumerate(results['results'], 1):
                        st.subheader(f"Result {i}")
                        display_product_card(product)
                        
                        # Add recommendation button
                        if st.button(f"Get Recommendations for Result {i}", key=f"rec_{i}"):
                            with st.spinner("Getting recommendations..."):
                                recs = get_recommendations(product['product_id'])
                                if recs and recs['recommendations']:
                                    st.success(f"🎯 Found {recs['total_recommendations']} recommendations!")
                                    
                                    for j, rec in enumerate(recs['recommendations'], 1):
                                        st.markdown(f"**Recommendation {j}:**")
                                        display_product_card(rec, show_score=False)
                                else:
                                    st.warning("No recommendations found.")
                    
                    # Show raw data option
                    with st.expander("View Raw Data"):
                        st.json(results)
                else:
                    st.warning("No results found. Try a different query.")
    
    # Tab 2: Model Performance
    with tab2:
        display_evaluation_metrics()
    
    # Tab 3: Dataset Statistics
    with tab3:
        try:
            response = requests.get(f"{BASE_URL}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Products", stats['total_products'])
                with col2:
                    st.metric("Brands", stats['brands'])
                with col3:
                    st.metric("Min Price", f"₹{stats['price_range']['min']:.0f}")
                with col4:
                    st.metric("Max Price", f"₹{stats['price_range']['max']:.0f}")
                
                # Categories breakdown
                st.subheader("Products by Category")
                categories_df = pd.DataFrame([
                    {"Category": cat, "Count": count} 
                    for cat, count in stats['categories'].items()
                ])
                st.bar_chart(categories_df.set_index("Category"))
                
            else:
                st.error("Failed to load statistics")
        except:
            st.warning("Could not load statistics")
    
    # Footer
    st.markdown("---")
    st.markdown("**API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)")

if __name__ == "__main__":
    main()
