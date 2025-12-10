import torch
import time
from sentence_transformers import SentenceTransformer

def synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()

def benchmark_semantic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Benchmarking Semantic Transformer on: {device}")
    semantic = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=str(device))
    semantic.eval()
    seq_lengths = [128, 256, 512]
    batch_size = 4
    iterations = 10
    print('\n' + '='*80)
    print(f"{'Seq Length':<12} | {'Semantic Transformer (ms)':<25} | {'Speedup':<10}")
    print('-'*80)
    for seq_len in seq_lengths:
        # create dummy sentences of approx length
        num_words = (seq_len * 5) // 6
        test_sentence = ' '.join(['word'] * num_words)
        sentences = [test_sentence] * batch_size
        # warmup
        warmup = 3 if device.type == 'cuda' else 1
        for _ in range(warmup):
            _ = semantic.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        synchronize_device(device)
        # measure
        start = time.time()
        for _ in range(iterations):
            _ = semantic.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        synchronize_device(device)
        elapsed = (time.time() - start) / iterations * 1000
        print(f"{seq_len:<12} | {elapsed:<25.2f} | {'-':<10}")
    print('='*80)
    print('✅ Semantic Transformer benchmark complete!')

if __name__ == '__main__':
    benchmark_semantic()
