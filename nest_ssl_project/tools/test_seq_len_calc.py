#!/usr/bin/env python
"""
测试 seq_len 计算在不同环境中的行为。
请在两个环境中分别运行此脚本，比较输出。
"""
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print()

# 测试参数（与配置一致）
sample_rate = 16000
window_size = 0.025  # 25ms
window_stride = 0.01  # 10ms
n_fft = 512
exact_pad = False

n_window_size = int(window_size * sample_rate)  # 400
n_window_stride = int(window_stride * sample_rate)  # 160
hop_length = n_window_stride  # 160

# stft_pad_amount
stft_pad_amount = (n_fft - hop_length) // 2 if exact_pad else None
pad_amount = stft_pad_amount * 2 if stft_pad_amount is not None else n_fft // 2 * 2

print(f"Parameters:")
print(f"  n_fft: {n_fft}")
print(f"  hop_length: {hop_length}")
print(f"  exact_pad: {exact_pad}")
print(f"  stft_pad_amount: {stft_pad_amount}")
print(f"  pad_amount: {pad_amount}")
print()

# 测试不同的 audio_len 值
test_lengths = [253040, 200000, 160000, 48000, 16000]

print("Testing get_seq_len calculation:")
print("-" * 60)

for audio_len in test_lengths:
    seq_len = torch.tensor([audio_len], dtype=torch.long)
    
    # 方法1: torch.floor_divide (NeMo 使用的方法)
    result1 = torch.floor_divide((seq_len + pad_amount - n_fft), hop_length)
    
    # 方法2: float division + floor (之前尝试的方法)
    seq_len_float = (seq_len.float() + pad_amount - n_fft) / hop_length
    result2 = seq_len_float.floor().to(dtype=torch.long)
    
    # 方法3: 直接整数除法
    result3 = (seq_len + pad_amount - n_fft) // hop_length
    
    print(f"audio_len={audio_len}:")
    print(f"  floor_divide: {result1.item()}")
    print(f"  float+floor:  {result2.item()}")
    print(f"  int division: {result3.item()}")
    
    if result1.item() != result2.item():
        print(f"  [DIFFERENCE] floor_divide != float+floor")
    if result1.item() != result3.item():
        print(f"  [DIFFERENCE] floor_divide != int division")
    print()

# 测试边界情况
print("Testing edge cases:")
print("-" * 60)

# 找到可能产生差异的值
for i in range(253000, 253100):
    seq_len = torch.tensor([i], dtype=torch.long)
    r1 = torch.floor_divide((seq_len + pad_amount - n_fft), hop_length).item()
    r2 = ((seq_len.float() + pad_amount - n_fft) / hop_length).floor().to(dtype=torch.long).item()
    
    if r1 != r2:
        print(f"audio_len={i}: floor_divide={r1}, float+floor={r2}")

print()
print("If you see any [DIFFERENCE] or edge case output above,")
print("it means the two methods produce different results in this environment.")

