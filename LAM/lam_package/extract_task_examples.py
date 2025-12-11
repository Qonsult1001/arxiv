#!/usr/bin/env python3
"""
Extract 5+ examples from each LongEmbed task
"""

from mteb import get_tasks
import json

LONGEMBED_TASKS = [
    "LEMBNarrativeQARetrieval",
    "LEMBQMSumRetrieval", 
    "LEMBWikimQARetrieval",
    "LEMBSummScreenFDRetrieval",
]

def extract_examples(task_name: str, num_examples: int = 5):
    """Extract multiple examples from a task."""
    print(f"\n{'='*80}")
    print(f"Extracting {num_examples} examples from {task_name}")
    print(f"{'='*80}")
    
    tasks = get_tasks(tasks=[task_name])
    if not tasks:
        return None
    
    task = tasks[0]
    task.load_data()
    
    # Get corpus
    corpus = task.corpus
    if isinstance(corpus, dict) and 'test' in corpus:
        corpus = corpus['test']
    elif isinstance(corpus, dict) and len(corpus) > 0:
        first_key = list(corpus.keys())[0]
        if isinstance(corpus[first_key], dict):
            corpus = corpus[first_key]
    
    # Get queries
    queries = task.queries
    if isinstance(queries, dict) and 'test' in queries:
        queries = queries['test']
    elif isinstance(queries, dict) and len(queries) > 0:
        first_key = list(queries.keys())[0]
        if isinstance(queries[first_key], dict):
            queries = queries[first_key]
    
    # Get evaluation data (relevance judgments)
    eval_data = None
    if hasattr(task, 'evaluation_data'):
        eval_data = task.evaluation_data
        if isinstance(eval_data, dict) and 'test' in eval_data:
            eval_data = eval_data['test']
    
    examples = []
    
    # Strategy 1: If we have evaluation data, use query-doc pairs from it
    if eval_data and isinstance(eval_data, dict):
        query_ids = list(eval_data.keys())[:num_examples]
        
        for q_id in query_ids:
            if q_id not in queries:
                continue
            
            query_text = str(queries[q_id])
            
            # Get relevant documents for this query
            relevant_docs = eval_data[q_id]
            if isinstance(relevant_docs, dict):
                # Get first relevant doc
                rel_doc_id = list(relevant_docs.keys())[0]
                if rel_doc_id in corpus:
                    doc = corpus[rel_doc_id]
                    title = doc.get('title', '')
                    body = doc.get('body', doc.get('text', ''))
                    
                    examples.append({
                        "query_id": q_id,
                        "query": query_text,
                        "relevant_doc_id": rel_doc_id,
                        "doc_title": title,
                        "doc_body_preview": body[:500] if body else "",
                        "doc_length": len(body) if body else 0
                    })
    
    # Strategy 2: If we don't have enough examples, pair queries with first docs
    if len(examples) < num_examples:
        query_ids = list(queries.keys())[:num_examples * 2]
        doc_ids = list(corpus.keys())[:num_examples * 2]
        
        for i in range(min(len(query_ids), len(doc_ids), num_examples - len(examples))):
            q_id = query_ids[i]
            doc_id = doc_ids[i]
            
            if q_id in queries and doc_id in corpus:
                query_text = str(queries[q_id])
                doc = corpus[doc_id]
                title = doc.get('title', '')
                body = doc.get('body', doc.get('text', ''))
                
                examples.append({
                    "query_id": q_id,
                    "query": query_text,
                    "relevant_doc_id": doc_id,
                    "doc_title": title,
                    "doc_body_preview": body[:500] if body else "",
                    "doc_length": len(body) if body else 0
                })
    
    return examples[:num_examples]

def format_examples_markdown(task_name: str, examples: list):
    """Format examples as markdown."""
    lines = []
    lines.append(f"### 📝 Examples ({len(examples)} examples)")
    lines.append("")
    
    for i, ex in enumerate(examples, 1):
        lines.append(f"#### Example {i}")
        lines.append("")
        lines.append(f"**Query:** `{ex['query'][:200]}{'...' if len(ex['query']) > 200 else ''}`")
        lines.append("")
        lines.append(f"**Query ID:** `{ex['query_id']}`")
        lines.append("")
        if ex.get('doc_title'):
            lines.append(f"**Document Title:** {ex['doc_title'][:150]}")
            lines.append("")
        lines.append(f"**Document ID:** `{ex['relevant_doc_id']}`")
        lines.append("")
        lines.append(f"**Document Length:** {ex['doc_length']:,} characters")
        lines.append("")
        lines.append("**Document Preview:**")
        lines.append("```")
        preview = ex['doc_body_preview']
        # Clean up HTML if present
        if preview.startswith('<html>'):
            preview = preview[:300] + "... [HTML content]"
        lines.append(preview[:400] + ("..." if len(ex['doc_body_preview']) > 400 else ""))
        lines.append("```")
        lines.append("")
        lines.append("**What to look for:**")
        lines.append(f"- Query asks: {ex['query'][:100]}...")
        lines.append(f"- Document should contain information relevant to answering this question")
        lines.append(f"- Semantic similarity between query intent and document content")
        lines.append("")
        if i < len(examples):
            lines.append("---")
            lines.append("")
    
    return "\n".join(lines)

# Main execution
all_examples = {}

for task_name in LONGEMBED_TASKS:
    try:
        examples = extract_examples(task_name, num_examples=7)  # Get 7, will use 5-7
        if examples:
            all_examples[task_name] = examples
            print(f"\n✅ Extracted {len(examples)} examples from {task_name}")
    except Exception as e:
        print(f"\n❌ Error extracting from {task_name}: {e}")
        import traceback
        traceback.print_exc()

# Generate markdown sections for each task
print("\n" + "="*80)
print("GENERATING MARKDOWN SECTIONS")
print("="*80)

for task_name, examples in all_examples.items():
    print(f"\n{task_name}:")
    markdown = format_examples_markdown(task_name, examples)
    
    # Save to file
    output_file = f"/workspace/LAM/lam_package/{task_name}_EXAMPLES.md"
    with open(output_file, 'w') as f:
        f.write(f"# {task_name} - Detailed Examples\n\n")
        f.write(markdown)
    
    print(f"  ✅ Saved to {output_file}")
    print(f"  📊 {len(examples)} examples extracted")

# Also create a combined file
combined_file = "/workspace/LAM/lam_package/ALL_LONGEMBED_EXAMPLES.md"
with open(combined_file, 'w') as f:
    f.write("# All LongEmbed Tasks - Detailed Examples\n\n")
    f.write("This document contains 5+ examples from each LongEmbed task.\n\n")
    f.write("---\n\n")
    
    for task_name, examples in all_examples.items():
        f.write(f"## {task_name}\n\n")
        f.write(format_examples_markdown(task_name, examples))
        f.write("\n\n---\n\n")

print(f"\n✅ Combined examples saved to: {combined_file}")


