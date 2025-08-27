#!/usr/bin/env python3
"""
SBERT Evaluation Script for Fashion Retail Analytics
Evaluates Precision, Recall, and Latency of the SBERT model
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import the search engine components
sys.path.append('/home/artisans15/projects/fashion_retail_analytics')
from scripts.search import semantic_search, df, model, embeddings

class SBERTEvaluator:
    """
    Comprehensive evaluator for SBERT model performance
    """
    
    def __init__(self):
        self.df = df
        self.model = model
        self.embeddings = embeddings
        self.test_queries = self._create_test_queries()
        self.results = {}
        
    def _create_test_queries(self) -> List[Dict]:
        """
        Create comprehensive test queries with ground truth relevance
        """
        # Analyze the dataset to understand product distribution
        footwear_products = self.df[self.df['product_category'] == 'Footwear']
        clothing_products = self.df[self.df['product_category'] == 'Clothing']
        accessories_products = self.df[self.df['product_category'] == 'Accessories']
        
        test_queries = [
            # Footwear queries
            {
                "query": "running shoes",
                "category": "Footwear",
                "relevant_products": footwear_products[
                    footwear_products['product_name'].str.contains('Running Shoes|Sneakers', case=False, na=False)
                ]['product_id'].tolist()[:10],
                "expected_type": "sneakers"
            },
            {
                "query": "party heels",
                "category": "Footwear", 
                "relevant_products": footwear_products[
                    footwear_products['product_name'].str.contains('Heels', case=False, na=False)
                ]['product_id'].tolist()[:8],
                "expected_type": "heels"
            },
            {
                "query": "leather boots",
                "category": "Footwear",
                "relevant_products": footwear_products[
                    (footwear_products['product_name'].str.contains('Boots', case=False, na=False)) &
                    (footwear_products['material'].str.contains('Leather', case=False, na=False))
                ]['product_id'].tolist()[:6],
                "expected_type": "boots"
            },
            {
                "query": "sandals",
                "category": "Footwear",
                "relevant_products": footwear_products[
                    footwear_products['product_name'].str.contains('Sandals', case=False, na=False)
                ]['product_id'].tolist()[:8],
                "expected_type": "sandals"
            },
            {
                "query": "slippers",
                "category": "Footwear",
                "relevant_products": footwear_products[
                    footwear_products['product_name'].str.contains('Slippers', case=False, na=False)
                ]['product_id'].tolist()[:8],
                "expected_type": "slippers"
            },
            
            # Clothing queries
            {
                "query": "cotton t-shirt",
                "category": "Clothing",
                "relevant_products": clothing_products[
                    (clothing_products['product_name'].str.contains('T-shirt', case=False, na=False)) &
                    (clothing_products['material'].str.contains('Cotton', case=False, na=False))
                ]['product_id'].tolist()[:5],
                "expected_type": "t-shirt"
            },
            {
                "query": "winter jacket",
                "category": "Clothing",
                "relevant_products": clothing_products[
                    clothing_products['product_name'].str.contains('Jacket|Hoodie', case=False, na=False)
                ]['product_id'].tolist()[:6],
                "expected_type": "jacket"
            },
            {
                "query": "formal dress",
                "category": "Clothing",
                "relevant_products": clothing_products[
                    clothing_products['product_name'].str.contains('Dress', case=False, na=False)
                ]['product_id'].tolist()[:4],
                "expected_type": "dress"
            },
            {
                "query": "wool sweater",
                "category": "Clothing",
                "relevant_products": clothing_products[
                    (clothing_products['product_name'].str.contains('Sweater', case=False, na=False)) &
                    (clothing_products['material'].str.contains('Wool', case=False, na=False))
                ]['product_id'].tolist()[:5],
                "expected_type": "sweater"
            },
            
            # Accessories queries
            {
                "query": "leather handbag",
                "category": "Accessories",
                "relevant_products": accessories_products[
                    (accessories_products['product_name'].str.contains('Handbag', case=False, na=False)) &
                    (accessories_products['material'].str.contains('Leather', case=False, na=False))
                ]['product_id'].tolist()[:5],
                "expected_type": "handbag"
            },
            {
                "query": "laptop backpack",
                "category": "Accessories",
                "relevant_products": accessories_products[
                    accessories_products['product_name'].str.contains('Backpack', case=False, na=False)
                ]['product_id'].tolist()[:6],
                "expected_type": "backpack"
            },
            {
                "query": "wrist watch",
                "category": "Accessories",
                "relevant_products": accessories_products[
                    accessories_products['product_name'].str.contains('Watch', case=False, na=False)
                ]['product_id'].tolist()[:8],
                "expected_type": "watch"
            },
            
            # Color-specific queries
            {
                "query": "black sneakers",
                "category": "Footwear",
                "relevant_products": footwear_products[
                    (footwear_products['product_name'].str.contains('Sneakers', case=False, na=False)) &
                    (footwear_products['color'].str.contains('Black', case=False, na=False))
                ]['product_id'].tolist()[:6],
                "expected_type": "sneakers"
            },
            {
                "query": "red dress",
                "category": "Clothing", 
                "relevant_products": clothing_products[
                    (clothing_products['product_name'].str.contains('Dress', case=False, na=False)) &
                    (clothing_products['color'].str.contains('Red', case=False, na=False))
                ]['product_id'].tolist()[:3],
                "expected_type": "dress"
            },
            
            # Brand-specific queries
            {
                "query": "nike shoes",
                "category": "Footwear",
                "relevant_products": footwear_products[
                    footwear_products['brand'].str.contains('Nike', case=False, na=False)
                ]['product_id'].tolist()[:8],
                "expected_type": "shoes"
            },
            {
                "query": "adidas sneakers", 
                "category": "Footwear",
                "relevant_products": footwear_products[
                    (footwear_products['brand'].str.contains('Adidas', case=False, na=False)) &
                    (footwear_products['product_name'].str.contains('Sneakers', case=False, na=False))
                ]['product_id'].tolist()[:5],
                "expected_type": "sneakers"
            }
        ]
        
        # Filter out queries with no relevant products
        valid_queries = [q for q in test_queries if len(q['relevant_products']) > 0]
        print(f"Created {len(valid_queries)} test queries with ground truth")
        return valid_queries
    
    def calculate_precision_recall_at_k(self, retrieved_ids: List[str], 
                                       relevant_ids: List[str], k: int) -> Tuple[float, float]:
        """
        Calculate Precision@K and Recall@K
        """
        if not retrieved_ids or not relevant_ids:
            return 0.0, 0.0
            
        retrieved_at_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        retrieved_set = set(retrieved_at_k)
        
        # True positives
        tp = len(relevant_set.intersection(retrieved_set))
        
        # Precision@K = TP / K
        precision_at_k = tp / k if k > 0 else 0.0
        
        # Recall@K = TP / total_relevant
        recall_at_k = tp / len(relevant_set) if len(relevant_set) > 0 else 0.0
        
        return precision_at_k, recall_at_k
    
    def measure_latency(self, query: str, num_runs: int = 10) -> Dict[str, float]:
        """
        Measure search latency with multiple runs for accuracy
        """
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.time()
            results = semantic_search(query, top_k=10)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'std_latency_ms': float(np.std(latencies))
        }
    
    def evaluate_query(self, query_info: Dict, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Evaluate a single query for precision, recall, and latency
        """
        query = query_info['query']
        relevant_ids = query_info['relevant_products']
        
        print(f"Evaluating query: '{query}' (Ground truth: {len(relevant_ids)} relevant products)")
        
        # Measure latency
        latency_metrics = self.measure_latency(query)
        
        # Get search results
        results = semantic_search(query, top_k=max(k_values))
        retrieved_ids = results['product_id'].tolist()
        
        # Calculate precision and recall for different k values
        precision_recall = {}
        for k in k_values:
            precision, recall = self.calculate_precision_recall_at_k(retrieved_ids, relevant_ids, k)
            precision_recall[f'precision@{k}'] = precision
            precision_recall[f'recall@{k}'] = recall
        
        # Calculate category accuracy
        expected_category = query_info['category']
        retrieved_categories = results['product_category'].tolist()
        category_accuracy = sum(1 for cat in retrieved_categories if cat == expected_category) / len(retrieved_categories)
        
        return {
            'query': query,
            'category': expected_category,
            'relevant_count': len(relevant_ids),
            'retrieved_count': len(retrieved_ids),
            'category_accuracy': category_accuracy,
            'latency_metrics': latency_metrics,
            **precision_recall
        }
    
    def run_full_evaluation(self) -> Dict:
        """
        Run complete evaluation on all test queries
        """
        print("=" * 60)
        print("SBERT MODEL EVALUATION - Fashion Retail Search Engine")
        print("=" * 60)
        print(f"Total test queries: {len(self.test_queries)}")
        print(f"Dataset size: {len(self.df)} products")
        print(f"Model: {model}")
        print("-" * 60)
        
        all_results = []
        category_results = defaultdict(list)
        
        for i, query_info in enumerate(self.test_queries, 1):
            print(f"\n[{i}/{len(self.test_queries)}]", end=" ")
            result = self.evaluate_query(query_info)
            all_results.append(result)
            category_results[result['category']].append(result)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(all_results)
        category_metrics = {cat: self._calculate_aggregate_metrics(results) 
                          for cat, results in category_results.items()}
        
        self.results = {
            'individual_results': all_results,
            'aggregate_metrics': aggregate_metrics,
            'category_metrics': category_metrics,
            'dataset_info': {
                'total_products': len(self.df),
                'categories': self.df['product_category'].value_counts().to_dict(),
                'brands': self.df['brand'].nunique(),
                'model_name': str(model)
            }
        }
        
        return self.results
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate aggregate metrics across all results
        """
        if not results:
            return {}
        
        k_values = [1, 3, 5, 10]
        aggregate = {}
        
        # Average precision and recall
        for k in k_values:
            precisions = [r[f'precision@{k}'] for r in results]
            recalls = [r[f'recall@{k}'] for r in results]
            
            aggregate[f'avg_precision@{k}'] = np.mean(precisions)
            aggregate[f'avg_recall@{k}'] = np.mean(recalls)
            aggregate[f'std_precision@{k}'] = np.std(precisions)
            aggregate[f'std_recall@{k}'] = np.std(recalls)
        
        # Average latency metrics
        latency_keys = ['mean_latency_ms', 'median_latency_ms', 'min_latency_ms', 'max_latency_ms']
        for key in latency_keys:
            values = [r['latency_metrics'][key] for r in results]
            aggregate[f'avg_{key}'] = np.mean(values)
            aggregate[f'std_{key}'] = np.std(values)
        
        # Category accuracy
        category_accuracies = [r['category_accuracy'] for r in results]
        aggregate['avg_category_accuracy'] = np.mean(category_accuracies)
        aggregate['std_category_accuracy'] = np.std(category_accuracies)
        
        # F1 scores
        for k in k_values:
            precisions = [r[f'precision@{k}'] for r in results]
            recalls = [r[f'recall@{k}'] for r in results]
            f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
            aggregate[f'avg_f1@{k}'] = np.mean(f1_scores)
            aggregate[f'std_f1@{k}'] = np.std(f1_scores)
        
        return aggregate
    
    def display_results(self):
        """
        Display comprehensive evaluation results
        """
        if not self.results:
            print("No results available. Run evaluation first.")
            return
        
        self._display_summary()
        self._display_detailed_metrics()
        self._display_category_breakdown()
        self._display_latency_analysis()
        self._create_visualizations()
        self._display_recommendations()
    
    def _display_summary(self):
        """
        Display executive summary of results
        """
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        agg = self.results['aggregate_metrics']
        
        print(f"Dataset: {self.results['dataset_info']['total_products']} products")
        print(f"Test Queries: {len(self.results['individual_results'])}")
        print(f"Categories: {list(self.results['dataset_info']['categories'].keys())}")
        
        print("\n📊 KEY PERFORMANCE INDICATORS")
        print("-" * 40)
        print(f"Precision@3:     {agg['avg_precision@3']:.3f} ± {agg['std_precision@3']:.3f}")
        print(f"Recall@3:        {agg['avg_recall@3']:.3f} ± {agg['std_recall@3']:.3f}")
        print(f"F1-Score@3:      {agg['avg_f1@3']:.3f} ± {agg['std_f1@3']:.3f}")
        print(f"Category Accuracy: {agg['avg_category_accuracy']:.3f} ± {agg['std_category_accuracy']:.3f}")
        print(f"Avg Latency:     {agg['avg_mean_latency_ms']:.1f}ms ± {agg['std_mean_latency_ms']:.1f}ms")
    
    def _display_detailed_metrics(self):
        """
        Display detailed precision/recall metrics
        """
        print("\n" + "=" * 60)
        print("DETAILED PRECISION & RECALL METRICS")
        print("=" * 60)
        
        agg = self.results['aggregate_metrics']
        
        print("\n📈 PRECISION @ K")
        print("-" * 30)
        for k in [1, 3, 5, 10]:
            print(f"P@{k:2d}: {agg[f'avg_precision@{k}']:.4f} ± {agg[f'std_precision@{k}']:.4f}")
        
        print("\n📈 RECALL @ K")
        print("-" * 30)
        for k in [1, 3, 5, 10]:
            print(f"R@{k:2d}: {agg[f'avg_recall@{k}']:.4f} ± {agg[f'std_recall@{k}']:.4f}")
        
        print("\n📈 F1-SCORE @ K")
        print("-" * 30)
        for k in [1, 3, 5, 10]:
            print(f"F1@{k:2d}: {agg[f'avg_f1@{k}']:.4f} ± {agg[f'std_f1@{k}']:.4f}")
    
    def _display_category_breakdown(self):
        """
        Display performance breakdown by category
        """
        print("\n" + "=" * 60)
        print("PERFORMANCE BY CATEGORY")
        print("=" * 60)
        
        for category, metrics in self.results['category_metrics'].items():
            print(f"\n🏷️  {category.upper()}")
            print("-" * 30)
            print(f"Precision@3: {metrics['avg_precision@3']:.3f}")
            print(f"Recall@3:    {metrics['avg_recall@3']:.3f}")
            print(f"F1@3:        {metrics['avg_f1@3']:.3f}")
            print(f"Latency:     {metrics['avg_mean_latency_ms']:.1f}ms")
    
    def _display_latency_analysis(self):
        """
        Display detailed latency analysis
        """
        print("\n" + "=" * 60)
        print("LATENCY ANALYSIS")
        print("=" * 60)
        
        agg = self.results['aggregate_metrics']
        
        print(f"\n⏱️  RESPONSE TIME STATISTICS")
        print("-" * 35)
        print(f"Mean:      {agg['avg_mean_latency_ms']:.1f}ms ± {agg['std_mean_latency_ms']:.1f}ms")
        print(f"Median:    {agg['avg_median_latency_ms']:.1f}ms ± {agg['std_median_latency_ms']:.1f}ms")
        print(f"Min:       {agg['avg_min_latency_ms']:.1f}ms")
        print(f"Max:       {agg['avg_max_latency_ms']:.1f}ms")
        
        # Performance classification
        avg_latency = agg['avg_mean_latency_ms']
        if avg_latency < 50:
            performance = "EXCELLENT (< 50ms)"
        elif avg_latency < 100:
            performance = "GOOD (50-100ms)"
        elif avg_latency < 200:
            performance = "ACCEPTABLE (100-200ms)" 
        else:
            performance = "NEEDS IMPROVEMENT (> 200ms)"
        
        print(f"\nPerformance Rating: {performance}")
    
    def _create_visualizations(self):
        """
        Create visualization charts for the results
        """
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SBERT Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Precision/Recall by K
        agg = self.results['aggregate_metrics']
        k_values = [1, 3, 5, 10]
        precisions = [agg[f'avg_precision@{k}'] for k in k_values]
        recalls = [agg[f'avg_recall@{k}'] for k in k_values]
        
        ax1.plot(k_values, precisions, 'o-', label='Precision', linewidth=2, markersize=8)
        ax1.plot(k_values, recalls, 's-', label='Recall', linewidth=2, markersize=8)
        ax1.set_xlabel('K')
        ax1.set_ylabel('Score')
        ax1.set_title('Precision & Recall @ K')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(k_values)
        
        # 2. Category Performance
        categories = list(self.results['category_metrics'].keys())
        cat_precisions = [self.results['category_metrics'][cat]['avg_precision@3'] for cat in categories]
        
        bars = ax2.bar(categories, cat_precisions, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_ylabel('Precision@3')
        ax2.set_title('Precision@3 by Category')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 3. Latency Distribution
        latencies = [r['latency_metrics']['mean_latency_ms'] for r in self.results['individual_results']]
        ax3.hist(latencies, bins=15, alpha=0.7, color='#96CEB4', edgecolor='black')
        ax3.set_xlabel('Latency (ms)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Latency Distribution')
        ax3.axvline(np.mean(latencies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(latencies):.1f}ms')
        ax3.legend()
        
        # 4. Individual Query Performance
        queries = [r['query'][:15] + '...' if len(r['query']) > 15 else r['query'] 
                  for r in self.results['individual_results']]
        f1_scores = [2 * r['precision@3'] * r['recall@3'] / (r['precision@3'] + r['recall@3']) 
                    if (r['precision@3'] + r['recall@3']) > 0 else 0 
                    for r in self.results['individual_results']]
        
        ax4.barh(range(len(queries)), f1_scores, color='#FFEAA7')
        ax4.set_yticks(range(len(queries)))
        ax4.set_yticklabels(queries, fontsize=8)
        ax4.set_xlabel('F1-Score@3')
        ax4.set_title('F1-Score@3 by Query')
        ax4.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path('/home/artisans15/projects/fashion_retail_analytics/evaluation_results')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'sbert_evaluation_charts.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualizations saved to: {output_dir / 'sbert_evaluation_charts.png'}")
        
        plt.show()
        
    def _display_recommendations(self):
        """
        Display recommendations for improvement
        """
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS FOR IMPROVEMENT")
        print("=" * 60)
        
        agg = self.results['aggregate_metrics']
        
        recommendations = []
        
        # Precision recommendations
        if agg['avg_precision@3'] < 0.5:
            recommendations.append("🔍 Low Precision@3 detected. Consider:")
            recommendations.append("   - Fine-tuning embeddings on fashion-specific data")
            recommendations.append("   - Improving query processing and facet extraction")
            recommendations.append("   - Adding more sophisticated ranking features")
        
        # Recall recommendations  
        if agg['avg_recall@3'] < 0.4:
            recommendations.append("📢 Low Recall@3 detected. Consider:")
            recommendations.append("   - Expanding the search to more candidates")
            recommendations.append("   - Using query expansion techniques")
            recommendations.append("   - Improving semantic similarity thresholds")
        
        # Latency recommendations
        if agg['avg_mean_latency_ms'] > 100:
            recommendations.append("⚡ High latency detected. Consider:")
            recommendations.append("   - Implementing approximate nearest neighbor search (FAISS)")
            recommendations.append("   - Caching frequent queries")
            recommendations.append("   - Using a smaller/faster model")
        
        # Category-specific recommendations
        worst_category = min(self.results['category_metrics'].items(), 
                           key=lambda x: x[1]['avg_precision@3'])
        recommendations.append(f"🏷️  {worst_category[0]} shows lowest performance. Consider:")
        recommendations.append(f"   - Category-specific model fine-tuning")
        recommendations.append(f"   - Improving {worst_category[0].lower()} product descriptions")
        
        if not recommendations:
            recommendations.append("✅ Performance looks good! Consider:")
            recommendations.append("   - A/B testing with different models")
            recommendations.append("   - Expanding evaluation with more diverse queries")
        
        for rec in recommendations:
            print(rec)
        
    def save_results(self, filepath: Optional[str] = None):
        """
        Save evaluation results to JSON file
        """
        if not self.results:
            print("No results to save. Run evaluation first.")
            return
        
        if filepath is None:
            output_dir = Path('/home/artisans15/projects/fashion_retail_analytics/evaluation_results')
            output_dir.mkdir(exist_ok=True)
            filepath = str(output_dir / 'sbert_evaluation_results.json')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy_types(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"💾 Results saved to: {filepath}")


def main():
    """
    Main function to run the SBERT evaluation
    """
    print("🚀 Starting SBERT Evaluation for Fashion Retail Analytics")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = SBERTEvaluator()
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Display results
    evaluator.display_results()
    
    # Save results
    evaluator.save_results()
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
