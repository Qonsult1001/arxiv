#!/usr/bin/env python3
"""
Comprehensive Analysis of MTEB LongEmbed Tasks
===============================================
This script analyzes each LongEmbed task to understand:
1. Data structure (queries, corpus, relevance judgments)
2. Query types and formats
3. Document structure and length
4. Evaluation metrics and scoring
5. What models need to optimize for
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from typing import Dict, List, Any
from mteb import get_tasks

# All LongEmbed tasks
LONGEMBED_TASKS = [
    "LEMBNarrativeQARetrieval",
    "LEMBQMSumRetrieval", 
    "LEMBWikimQARetrieval",
    "LEMBSummScreenFDRetrieval",
]

def analyze_task(task_name: str) -> Dict[str, Any]:
    """Analyze a single LongEmbed task in detail."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {task_name}")
    print(f"{'='*80}")
    
    # Get task
    tasks = get_tasks(tasks=[task_name])
    if not tasks:
        return {"error": "Task not found"}
    
    task = tasks[0]
    
    # Load data
    print("\n📚 Loading task data...")
    task.load_data()
    
    analysis = {
        "task_name": task_name,
        "task_type": str(type(task)),
        "metadata": {},
        "corpus": {},
        "queries": {},
        "relevance": {},
        "evaluation": {},
        "examples": {}
    }
    
    # Get metadata
    if hasattr(task, 'metadata'):
        analysis["metadata"] = {
            "description": getattr(task.metadata, 'description', 'N/A'),
            "homepage": getattr(task.metadata, 'homepage', 'N/A'),
            "languages": getattr(task.metadata, 'languages', []),
        }
    
    # Analyze corpus structure
    print("\n📄 Analyzing corpus structure...")
    corpus = task.corpus
    if corpus:
        # Handle nested structure
        if isinstance(corpus, dict) and 'test' in corpus:
            corpus = corpus['test']
        elif isinstance(corpus, dict) and len(corpus) > 0:
            first_key = list(corpus.keys())[0]
            if isinstance(corpus[first_key], dict):
                corpus = corpus[first_key]
        
        corpus_ids = list(corpus.keys())[:10]  # Sample 10 docs
        
        # Analyze document structure
        sample_doc = corpus[corpus_ids[0]] if corpus_ids else None
        if sample_doc:
            analysis["corpus"] = {
                "total_docs": len(corpus),
                "sample_keys": list(sample_doc.keys()),
                "has_title": 'title' in sample_doc,
                "has_body": 'body' in sample_doc,
                "has_text": 'text' in sample_doc,
                "sample_doc_id": corpus_ids[0],
                "sample_doc": {
                    "title": sample_doc.get('title', '')[:200] if 'title' in sample_doc else None,
                    "body_length": len(sample_doc.get('body', sample_doc.get('text', ''))) if ('body' in sample_doc or 'text' in sample_doc) else 0,
                    "body_preview": (sample_doc.get('body', sample_doc.get('text', ''))[:500] if ('body' in sample_doc or 'text' in sample_doc) else None)
                }
            }
            
            # Calculate document lengths
            doc_lengths = []
            for doc_id in corpus_ids[:100]:  # Sample 100 docs
                doc = corpus[doc_id]
                text = f"{doc.get('title', '')} {doc.get('body', doc.get('text', ''))}".strip()
                doc_lengths.append(len(text))
            
            if doc_lengths:
                analysis["corpus"]["length_stats"] = {
                    "min_chars": min(doc_lengths),
                    "max_chars": max(doc_lengths),
                    "avg_chars": sum(doc_lengths) / len(doc_lengths),
                    "median_chars": sorted(doc_lengths)[len(doc_lengths)//2]
                }
    
    # Analyze queries
    print("\n🔍 Analyzing queries...")
    queries = task.queries
    if queries:
        # Handle nested structure
        if isinstance(queries, dict) and 'test' in queries:
            queries = queries['test']
        elif isinstance(queries, dict) and len(queries) > 0:
            first_key = list(queries.keys())[0]
            if isinstance(queries[first_key], dict):
                queries = queries[first_key]
        
        query_ids = list(queries.keys())[:10]  # Sample 10 queries
        
        analysis["queries"] = {
            "total_queries": len(queries),
            "sample_queries": []
        }
        
        for q_id in query_ids[:5]:  # Show 5 examples
            query_text = str(queries[q_id])
            analysis["queries"]["sample_queries"].append({
                "id": q_id,
                "text": query_text[:300],
                "length": len(query_text)
            })
    
    # Analyze relevance judgments
    print("\n📊 Analyzing relevance judgments...")
    if hasattr(task, 'evaluation_data'):
        eval_data = task.evaluation_data
        if eval_data:
            # Get test split
            if isinstance(eval_data, dict) and 'test' in eval_data:
                eval_data = eval_data['test']
            
            # Sample relevance pairs
            sample_pairs = []
            if isinstance(eval_data, list):
                for pair in eval_data[:5]:
                    sample_pairs.append({
                        "query_id": pair.get('query_id', pair.get('query', 'N/A')),
                        "corpus_id": pair.get('corpus_id', pair.get('corpus', 'N/A')),
                        "score": pair.get('score', pair.get('relevance', 'N/A'))
                    })
            elif isinstance(eval_data, dict):
                # Try to extract relevance pairs
                for q_id in list(eval_data.keys())[:3]:
                    rel_docs = eval_data[q_id]
                    if isinstance(rel_docs, dict):
                        for doc_id, score in list(rel_docs.items())[:2]:
                            sample_pairs.append({
                                "query_id": q_id,
                                "corpus_id": doc_id,
                                "score": score
                            })
            
            analysis["relevance"] = {
                "has_relevance": True,
                "sample_pairs": sample_pairs,
                "total_pairs": len(eval_data) if isinstance(eval_data, list) else "N/A"
            }
    
    # Get evaluation splits and metrics
    print("\n📈 Analyzing evaluation metrics...")
    if hasattr(task, 'eval_splits'):
        analysis["evaluation"]["splits"] = task.eval_splits
    
    if hasattr(task, 'k_values'):
        analysis["evaluation"]["k_values"] = task.k_values
    
    # Check what metrics are used
    # MTEB typically uses: ndcg_at_k, map_at_k, recall_at_k, precision_at_k, mrr_at_k
    analysis["evaluation"]["metrics"] = [
        "ndcg_at_1", "ndcg_at_3", "ndcg_at_5", "ndcg_at_10",
        "map_at_1", "map_at_3", "map_at_5", "map_at_10",
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "precision_at_1", "precision_at_3", "precision_at_5", "precision_at_10",
        "mrr_at_1", "mrr_at_3", "mrr_at_5", "mrr_at_10"
    ]
    
    # Main score (typically ndcg_at_10)
    analysis["evaluation"]["main_score"] = "ndcg_at_10"
    
    # Create examples
    print("\n📝 Creating examples...")
    if corpus_ids and query_ids:
        example_doc = corpus[corpus_ids[0]]
        example_query = queries[query_ids[0]]
        
        analysis["examples"] = {
            "document": {
                "id": corpus_ids[0],
                "title": example_doc.get('title', 'N/A')[:200],
                "body_preview": (example_doc.get('body', example_doc.get('text', ''))[:500] if ('body' in example_doc or 'text' in example_doc) else 'N/A')
            },
            "query": {
                "id": query_ids[0],
                "text": str(example_query)[:300]
            }
        }
    
    print(f"\n✅ Analysis complete for {task_name}")
    return analysis

def generate_report(analyses: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive markdown report."""
    report = []
    report.append("# 📊 MTEB LongEmbed Tasks - Complete Breakdown")
    report.append("")
    report.append("This document provides a comprehensive breakdown of each LongEmbed task in MTEB.")
    report.append("Use this to understand what each task tests and how to optimize your model.")
    report.append("")
    report.append("---")
    report.append("")
    
    for analysis in analyses:
        if "error" in analysis:
            continue
        
        task_name = analysis["task_name"]
        report.append(f"## {task_name}")
        report.append("")
        
        # Task description
        if analysis.get("metadata", {}).get("description"):
            report.append(f"**Description:** {analysis['metadata']['description']}")
            report.append("")
        
        # Corpus structure
        if analysis.get("corpus"):
            corpus_info = analysis["corpus"]
            report.append("### 📄 Corpus (Documents)")
            report.append("")
            report.append(f"- **Total documents:** {corpus_info.get('total_docs', 'N/A')}")
            report.append(f"- **Document structure:** {', '.join(corpus_info.get('sample_keys', []))}")
            report.append(f"- **Has title:** {corpus_info.get('has_title', False)}")
            report.append(f"- **Has body/text:** {corpus_info.get('has_body', False) or corpus_info.get('has_text', False)}")
            report.append("")
            
            if "length_stats" in corpus_info:
                stats = corpus_info["length_stats"]
                report.append("**Document Length Statistics:**")
                report.append(f"- Min: {stats['min_chars']:,} characters")
                report.append(f"- Max: {stats['max_chars']:,} characters")
                report.append(f"- Average: {stats['avg_chars']:,.0f} characters")
                report.append(f"- Median: {stats['median_chars']:,} characters")
                report.append("")
            
            if "sample_doc" in corpus_info:
                sample = corpus_info["sample_doc"]
                report.append("**Example Document:**")
                if sample.get("title"):
                    report.append(f"- **Title:** {sample['title']}")
                if sample.get("body_preview"):
                    report.append(f"- **Body preview:** {sample['body_preview']}...")
                report.append("")
        
        # Queries
        if analysis.get("queries"):
            query_info = analysis["queries"]
            report.append("### 🔍 Queries")
            report.append("")
            report.append(f"- **Total queries:** {query_info.get('total_queries', 'N/A')}")
            report.append("")
            report.append("**Example Queries:**")
            for i, query in enumerate(query_info.get("sample_queries", [])[:5], 1):
                report.append(f"{i}. `{query['text']}` ({query['length']} chars)")
            report.append("")
        
        # Relevance judgments
        if analysis.get("relevance"):
            rel_info = analysis["relevance"]
            report.append("### 📊 Relevance Judgments")
            report.append("")
            if rel_info.get("has_relevance"):
                report.append("**Relevance Format:**")
                for pair in rel_info.get("sample_pairs", [])[:3]:
                    report.append(f"- Query `{pair['query_id']}` → Doc `{pair['corpus_id']}` (score: {pair['score']})")
                report.append("")
        
        # Evaluation metrics
        if analysis.get("evaluation"):
            eval_info = analysis["evaluation"]
            report.append("### 📈 Evaluation Metrics")
            report.append("")
            report.append(f"- **Main score:** {eval_info.get('main_score', 'ndcg_at_10')}")
            report.append(f"- **K values:** {eval_info.get('k_values', [1, 3, 5, 10, 20, 100, 1000])}")
            report.append("")
            report.append("**All Metrics Used:**")
            for metric in eval_info.get("metrics", []):
                report.append(f"- `{metric}`")
            report.append("")
            report.append("**What This Means:**")
            report.append("- **nDCG@k:** Normalized Discounted Cumulative Gain at k - measures ranking quality")
            report.append("- **MAP@k:** Mean Average Precision at k - measures precision across all relevant docs")
            report.append("- **Recall@k:** Fraction of relevant docs found in top k")
            report.append("- **Precision@k:** Fraction of top k docs that are relevant")
            report.append("- **MRR@k:** Mean Reciprocal Rank - position of first relevant doc")
            report.append("")
            report.append("**Key Insight:** The main score is typically **nDCG@10**, which means:")
            report.append("1. Top 10 results matter most")
            report.append("2. Position matters (higher rank = better score)")
            report.append("3. Need to rank relevant documents higher than irrelevant ones")
            report.append("")
        
        # Examples
        if analysis.get("examples"):
            examples = analysis["examples"]
            report.append("### 📝 Complete Example")
            report.append("")
            if "query" in examples:
                report.append(f"**Query:** `{examples['query']['text']}`")
                report.append("")
            if "document" in examples:
                doc = examples["document"]
                report.append("**Relevant Document:**")
                if doc.get("title"):
                    report.append(f"- **Title:** {doc['title']}")
                if doc.get("body_preview"):
                    report.append(f"- **Body:** {doc['body_preview']}...")
                report.append("")
        
        # Optimization tips
        report.append("### 🎯 Optimization Tips")
        report.append("")
        report.append("**What the task tests:**")
        if "Narrative" in task_name:
            report.append("- Story comprehension and narrative understanding")
            report.append("- Ability to retrieve relevant story passages for questions")
            report.append("- Long document understanding (50K+ words)")
        elif "QMSum" in task_name:
            report.append("- Meeting summarization and key point extraction")
            report.append("- Retrieval of relevant meeting segments")
            report.append("- Understanding of conversational context")
        elif "Wikim" in task_name:
            report.append("- Multi-hop Wikipedia question answering")
            report.append("- Cross-document reasoning")
            report.append("- Wikipedia article understanding")
        elif "SummScreen" in task_name:
            report.append("- TV show summarization")
            report.append("- Dialogue and plot understanding")
            report.append("- Character and event retrieval")
        
        report.append("")
        report.append("**How to optimize:**")
        report.append("1. **Query encoding:** Use instruction-tuned prompts if your model supports them")
        report.append("2. **Document encoding:** Use streaming for long documents (no truncation)")
        report.append("3. **Similarity:** Use cosine similarity on normalized embeddings")
        report.append("4. **Focus on top-10:** nDCG@10 means ranking quality in top 10 matters most")
        report.append("5. **Semantic understanding:** These are semantic tasks, not exact matching")
        report.append("")
        
        report.append("---")
        report.append("")
    
    # Add comparison table
    report.append("## 📊 Task Comparison")
    report.append("")
    report.append("| Task | Documents | Queries | Main Metric | Key Challenge |")
    report.append("|------|-----------|---------|-------------|---------------|")
    
    for analysis in analyses:
        if "error" in analysis:
            continue
        
        task_name = analysis["task_name"]
        corpus_info = analysis.get("corpus", {})
        query_info = analysis.get("queries", {})
        eval_info = analysis.get("evaluation", {})
        
        num_docs = corpus_info.get("total_docs", "N/A")
        num_queries = query_info.get("total_queries", "N/A")
        main_metric = eval_info.get("main_score", "ndcg_at_10")
        
        if "Narrative" in task_name:
            challenge = "Long narrative comprehension"
        elif "QMSum" in task_name:
            challenge = "Meeting context understanding"
        elif "Wikim" in task_name:
            challenge = "Multi-hop reasoning"
        elif "SummScreen" in task_name:
            challenge = "Dialogue understanding"
        else:
            challenge = "Long context retrieval"
        
        report.append(f"| {task_name} | {num_docs} | {num_queries} | {main_metric} | {challenge} |")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 🏆 Competitive Scores")
    report.append("")
    report.append("| Model | Params | NarrativeQA | QMSum | WikimQA | SummScreen |")
    report.append("|-------|--------|-------------|-------|---------|------------|")
    report.append("| E5-Mistral-7B (SOTA) | 7B | 44.6 | 43.6 | 82.0 | 96.8 |")
    report.append("| BGE-M3 | 568M | 45.8 | 35.5 | 78.0 | 94.0 |")
    report.append("| OpenAI Ada-002 | ? | 41.1 | 40.0 | 80.1 | 91.8 |")
    report.append("| Jina-Embeddings-v2-Base | 137M | 37.9 | 38.9 | 74.0 | 93.5 |")
    report.append("| Nomic-Embed-Text-v1 | 137M | 41.2 | 36.7 | 73.8 | 93.0 |")
    report.append("| M2-BERT-32k (LoCo SOTA) | 80M | ~60.0* | High* | - | High* |")
    report.append("")
    report.append("*Note: Scores are nDCG@10 percentages*")
    report.append("")
    
    return "\n".join(report)

def main():
    """Main analysis function."""
    print("="*80)
    print("MTEB LongEmbed Tasks - Comprehensive Analysis")
    print("="*80)
    
    all_analyses = []
    
    for task_name in LONGEMBED_TASKS:
        try:
            analysis = analyze_task(task_name)
            all_analyses.append(analysis)
        except Exception as e:
            print(f"\n❌ Error analyzing {task_name}: {e}")
            import traceback
            traceback.print_exc()
            all_analyses.append({"task_name": task_name, "error": str(e)})
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    report = generate_report(all_analyses)
    
    # Save report
    output_path = Path(__file__).parent / "LONGEMBED_TASKS_BREAKDOWN.md"
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {output_path}")
    print(f"\n📊 Analyzed {len([a for a in all_analyses if 'error' not in a])}/{len(LONGEMBED_TASKS)} tasks")
    
    # Also print summary
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    for analysis in all_analyses:
        if "error" not in analysis:
            task_name = analysis["task_name"]
            corpus_info = analysis.get("corpus", {})
            query_info = analysis.get("queries", {})
            print(f"\n{task_name}:")
            print(f"  Documents: {corpus_info.get('total_docs', 'N/A')}")
            print(f"  Queries: {query_info.get('total_queries', 'N/A')}")
            print(f"  Main metric: {analysis.get('evaluation', {}).get('main_score', 'ndcg_at_10')}")

if __name__ == "__main__":
    main()

