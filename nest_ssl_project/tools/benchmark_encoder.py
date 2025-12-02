#!/usr/bin/env python
"""
Benchmark script to compare performance between NeMo's original encoder
and our local implementation.
"""

import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

def benchmark_encoder(encoder, input_signal, length, num_warmup=5, num_iter=50, device='cuda'):
    """Benchmark encoder forward pass."""
    encoder = encoder.to(device)
    encoder.eval()
    input_signal = input_signal.to(device)
    length = length.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = encoder(audio_signal=input_signal, length=length)
    
    # Synchronize GPU
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iter):
            _ = encoder(audio_signal=input_signal, length=length)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iter
    return avg_time

def profile_encoder(encoder, input_signal, length, device='cuda'):
    """Profile encoder using PyTorch profiler."""
    encoder = encoder.to(device)
    encoder.eval()
    input_signal = input_signal.to(device)
    length = length.to(device)
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            _ = encoder(audio_signal=input_signal, length=length)
    
    return prof

def main():
    parser = argparse.ArgumentParser(description='Benchmark encoder performance')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use for benchmarking')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for benchmarking')
    parser.add_argument('--seq_length', type=int, default=1500,
                       help='Sequence length (number of frames)')
    parser.add_argument('--feat_dim', type=int, default=80,
                       help='Feature dimension')
    parser.add_argument('--num_iter', type=int, default=50,
                       help='Number of iterations for benchmarking')
    parser.add_argument('--profile', action='store_true',
                       help='Use PyTorch profiler for detailed analysis')
    parser.add_argument('--compare', action='store_true',
                       help='Compare NeMo vs local implementation')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Create dummy input
    input_signal = torch.randn(args.batch_size, args.feat_dim, args.seq_length)
    length = torch.full((args.batch_size,), args.seq_length, dtype=torch.int64)
    
    print(f"\nInput shape: {input_signal.shape}")
    print(f"Length shape: {length.shape}")
    
    if args.compare:
        # Test NeMo's encoder
        try:
            from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder as NeMoEncoder
            print("\n" + "="*60)
            print("Testing NeMo's Original Encoder")
            print("="*60)
            
            nemo_encoder = NeMoEncoder(
                feat_in=args.feat_dim,
                n_layers=17,
                d_model=512,
                subsampling='dw_striding',
                subsampling_factor=8,
                subsampling_conv_channels=256,
            )
            
            nemo_time = benchmark_encoder(nemo_encoder, input_signal, length, 
                                         num_iter=args.num_iter, device=device)
            print(f"NeMo Encoder Average Time: {nemo_time*1000:.2f} ms")
            
            if args.profile:
                print("\nProfiling NeMo encoder...")
                nemo_prof = profile_encoder(nemo_encoder, input_signal, length, device=device)
                print(nemo_prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        except Exception as e:
            print(f"Failed to load NeMo encoder: {e}")
            nemo_time = None
    
    # Test local encoder
    try:
        from modules.conformer_encoder import ConformerEncoder as LocalEncoder
        print("\n" + "="*60)
        print("Testing Local Encoder")
        print("="*60)
        
        local_encoder = LocalEncoder(
            feat_in=args.feat_dim,
            n_layers=17,
            d_model=512,
            subsampling='dw_striding',
            subsampling_factor=8,
            subsampling_conv_channels=256,
        )
        
        local_time = benchmark_encoder(local_encoder, input_signal, length,
                                      num_iter=args.num_iter, device=device)
        print(f"Local Encoder Average Time: {local_time*1000:.2f} ms")
        
        if args.profile:
            print("\nProfiling local encoder...")
            local_prof = profile_encoder(local_encoder, input_signal, length, device=device)
            print(local_prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        if args.compare and nemo_time is not None:
            print("\n" + "="*60)
            print("Performance Comparison")
            print("="*60)
            speedup = nemo_time / local_time
            if speedup > 1:
                print(f"NeMo is {speedup:.2f}x faster")
            else:
                print(f"Local is {1/speedup:.2f}x faster")
            print(f"Time difference: {(local_time - nemo_time)*1000:.2f} ms")
            
    except Exception as e:
        print(f"Failed to load local encoder: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

