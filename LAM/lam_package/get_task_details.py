#!/usr/bin/env python3
"""
Get detailed examples from MTEB LongEmbed tasks
"""

from mteb import get_tasks

for task_name in ['LEMBNarrativeQARetrieval', 'LEMBQMSumRetrieval', 'LEMBWikimQARetrieval', 'LEMBSummScreenFDRetrieval']:
    print(f'\n{"="*80}')
    print(f'{task_name}')
    print(f'{"="*80}')
    tasks = get_tasks(tasks=[task_name])
    if tasks:
        task = tasks[0]
        task.load_data()
        
        # Get corpus
        corpus = task.corpus
        if isinstance(corpus, dict) and 'test' in corpus:
            corpus = corpus['test']
        
        # Get queries
        queries = task.queries
        if isinstance(queries, dict) and 'test' in queries:
            queries = queries['test']
        
        # Get evaluation data
        if hasattr(task, 'evaluation_data'):
            eval_data = task.evaluation_data
            if isinstance(eval_data, dict) and 'test' in eval_data:
                eval_data = eval_data['test']
            
            # Show first query and its relevant docs
            if isinstance(eval_data, dict):
                first_qid = list(eval_data.keys())[0]
                print(f'\nQuery ID: {first_qid}')
                if first_qid in queries:
                    print(f'Query: {queries[first_qid][:200]}')
                print(f'Relevant docs: {list(eval_data[first_qid].keys())[:5]}')
        
        # Show document example
        if corpus:
            first_doc_id = list(corpus.keys())[0]
            doc = corpus[first_doc_id]
            print(f'\nDocument ID: {first_doc_id}')
            if 'title' in doc:
                print(f'Title: {doc["title"][:100]}')
            if 'body' in doc:
                print(f'Body length: {len(doc["body"])} chars')
                print(f'Body preview: {doc["body"][:300]}...')
            elif 'text' in doc:
                print(f'Text length: {len(doc["text"])} chars')
                print(f'Text preview: {doc["text"][:300]}...')


