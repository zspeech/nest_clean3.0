#!/usr/bin/env python
"""
调试 seq_len 不匹配问题。
直接测试 torch.floor_divide 在不同环境中的行为。
"""
import torch
import sys

print("="*80)
print("PyTorch Version and Environment Info")
print("="*80)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
print()

# 测试参数（与实际配置一致）
n_fft = 512
hop_length = 160
exact_pad = False

# 计算 pad_amount
stft_pad_amount = (n_fft - hop_length) // 2 if exact_pad else None
pad_amount = stft_pad_amount * 2 if stft_pad_amount is not None else n_fft // 2 * 2

print(f"Parameters:")
print(f"  n_fft: {n_fft}")
print(f"  hop_length: {hop_length}")
print(f"  exact_pad: {exact_pad}")
print(f"  pad_amount: {pad_amount}")
print()

# 测试实际使用的 audio_len 值
# 根据比较结果，NeMo seq_len=1249, nest seq_len=1248
# 反推：如果 seq_len = floor_divide((audio_len + pad_amount - n_fft), hop_length)
# 那么：audio_len = seq_len * hop_length + n_fft - pad_amount
# NeMo: 1249 * 160 + 512 - 512 = 199840
# nest: 1248 * 160 + 512 - 512 = 199680

test_audio_lens = [253040, 200000, 199840, 199680, 199840 - 160, 199840 + 160]

print("="*80)
print("Testing get_seq_len calculation with actual values")
print("="*80)

for audio_len in test_audio_lens:
    seq_len = torch.tensor([audio_len], dtype=torch.long)
    
    # NeMo 的方法
    result_nemo = torch.floor_divide((seq_len + pad_amount - n_fft), hop_length)
    
    # 其他可能的方法
    result_float_floor = ((seq_len.float() + pad_amount - n_fft) / hop_length).floor().to(torch.long)
    result_int_div = (seq_len + pad_amount - n_fft) // hop_length
    
    print(f"audio_len={audio_len}:")
    print(f"  NeMo (floor_divide):     {result_nemo.item()}")
    print(f"  float+floor:             {result_float_floor.item()}")
    print(f"  int division:            {result_int_div.item()}")
    
    if result_nemo.item() != result_float_floor.item():
        print(f"  [DIFF] floor_divide != float+floor")
    if result_nemo.item() != result_int_div.item():
        print(f"  [DIFF] floor_divide != int division")
    
    # 反推验证
    if result_nemo.item() == 1249:
        print(f"  [MATCH] This gives NeMo's seq_len=1249")
    if result_nemo.item() == 1248:
        print(f"  [MATCH] This gives nest's seq_len=1248")
    print()

# 测试边界情况：找到所有可能产生 1248 vs 1249 差异的值
print("="*80)
print("Finding audio_len values that produce seq_len=1248 vs 1249")
print("="*80)

found_diff = False
for audio_len in range(199600, 200000, 1):
    seq_len = torch.tensor([audio_len], dtype=torch.long)
    r1 = torch.floor_divide((seq_len + pad_amount - n_fft), hop_length).item()
    
    if r1 == 1248 or r1 == 1249:
        print(f"audio_len={audio_len}: seq_len={r1}")
        if r1 == 1249:
            found_diff = True

if not found_diff:
    print("No audio_len found that produces seq_len=1249")
    print("This suggests the issue is elsewhere.")

print()
print("="*80)
print("Recommendations")
print("="*80)
print("1. Run this script in BOTH environments (Windows and Linux)")
print("2. Compare the outputs - if they differ, PyTorch versions are different")
print("3. If outputs are the same but seq_len still differs, check:")
print("   - Is the latest code pulled in Linux?")
print("   - Are the preprocessor configurations identical?")
print("   - Is exact_pad set differently?")

