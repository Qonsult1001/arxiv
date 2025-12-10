import torch
import time
from train_6layer_deltanet_2 import DeltaNetPure6Layer

def synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()

def benchmark_deltanet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Benchmarking DeltaNet on: {device}")
    config = {
        "teacher_model": "/workspace/LAM/all-MiniLM-L6-v2",
        "num_linear_layers": 6,
        "d_model": 384,
        "num_heads": 12,
        "fast_decay_init": 0.9,
        "slow_decay_init": 0.999,
        "use_kernel_blending": False,
    }
    deltanet = DeltaNetPure6Layer(
        teacher_model_name=config["teacher_model"],
        num_linear_layers=config["num_linear_layers"],
        config=config,
    ).to(device)
    deltanet.eval()
    seq_lengths = [128, 256, 512]
    batch_size = 4
    iterations = 10
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"])
    print('\n' + '='*80)
    print(f"{'Seq Length':<12} | {'DeltaNet (ms)':<20}")
    print('-'*80)
    for seq_len in seq_lengths:
        num_words = (seq_len * 5) // 6
        test_sentence = ' '.join(['word'] * num_words)
        sentences = [test_sentence] * batch_size
        tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=seq_len).to(device)
        # warmup
        warmup = 3 if device.type == 'cuda' else 1
        for _ in range(warmup):
            _ = deltanet.encode(tokens['input_ids'], tokens['attention_mask'])
        synchronize_device(device)
        start = time.time()
        for _ in range(iterations):
            _ = deltanet.encode(tokens['input_ids'], tokens['attention_mask'])
        synchronize_device(device)
        elapsed = (time.time() - start) / iterations * 1000
        print(f"{seq_len:<12} | {elapsed:<20.2f}")
    print('='*80)
    print('✅ DeltaNet benchmark complete!')

if __name__ == '__main__':
    benchmark_deltanet()
