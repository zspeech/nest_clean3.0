#!/usr/bin/env python
"""
详细诊断 FilterbankFeatures 差异来源。
逐步比较每个计算步骤的输出。
"""
import sys
from pathlib import Path
import torch
import pickle

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nemo_dir', type=str, default='./saved_nemo_outputs')
    parser.add_argument('--nest_dir', type=str, default='./saved_nest_outputs')
    parser.add_argument('--step', type=int, default=None)
    args = parser.parse_args()
    
    nemo_dir = Path(args.nemo_dir).resolve()
    nest_dir = Path(args.nest_dir).resolve()
    
    # Auto-detect step
    if args.step is None:
        nemo_steps = [d.name for d in nemo_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
        nest_steps = [d.name for d in nest_dir.iterdir() if d.is_dir() and d.name.startswith('step_')]
        common_steps = set(nemo_steps) & set(nest_steps)
        if common_steps:
            step_name = sorted(common_steps, key=lambda x: int(x.split('_')[1]))[0]
            args.step = int(step_name.split('_')[1])
        else:
            print(f"No common steps found. NeMo: {nemo_steps}, nest: {nest_steps}")
            return
    
    nemo_step_dir = nemo_dir / f'step_{args.step}'
    nest_step_dir = nest_dir / f'step_{args.step}'
    
    print(f"Loading layer outputs from step {args.step}...")
    print(f"NeMo dir: {nemo_step_dir}")
    print(f"nest dir: {nest_step_dir}")
    
    with open(nemo_step_dir / 'layer_outputs.pkl', 'rb') as f:
        nemo_layers = pickle.load(f)
    
    with open(nest_step_dir / 'layer_outputs.pkl', 'rb') as f:
        nest_layers = pickle.load(f)
    
    print("\n" + "="*80)
    print("FEATURIZER 详细分析")
    print("="*80)
    
    # 检查 preprocessor.featurizer 的输入和输出
    featurizer_name = 'preprocessor.featurizer'
    
    nemo_feat = nemo_layers.get(featurizer_name, {})
    nest_feat = nest_layers.get(featurizer_name, {})
    
    if not nemo_feat or not nest_feat:
        print(f"Warning: {featurizer_name} not found in one of the outputs")
        # 尝试其他名称
        for name in nemo_layers.keys():
            if 'featurizer' in name.lower():
                print(f"  Found: {name}")
        return
    
    # 比较输入
    print("\n--- Forward Inputs ---")
    nemo_inputs = nemo_feat.get('forward_inputs', [])
    nest_inputs = nest_feat.get('forward_inputs', [])
    
    if nemo_inputs and nest_inputs:
        for i, (ni, ne) in enumerate(zip(nemo_inputs, nest_inputs)):
            if isinstance(ni, torch.Tensor) and isinstance(ne, torch.Tensor):
                if ni.shape == ne.shape:
                    diff = (ni.float() - ne.float()).abs()
                    print(f"Input[{i}]: shape={list(ni.shape)}")
                    print(f"  Max diff: {diff.max().item():.6e}")
                    print(f"  Mean diff: {diff.mean().item():.6e}")
                    if diff.max().item() > 1e-6:
                        print(f"  [MISMATCH] Inputs are different!")
                    else:
                        print(f"  [OK] Inputs match")
                else:
                    print(f"Input[{i}]: Shape mismatch {ni.shape} vs {ne.shape}")
    
    # 比较输出
    print("\n--- Forward Outputs ---")
    nemo_outputs = nemo_feat.get('forward_outputs', [])
    nest_outputs = nest_feat.get('forward_outputs', [])
    
    if isinstance(nemo_outputs, (list, tuple)) and isinstance(nest_outputs, (list, tuple)):
        for i, (no, neo) in enumerate(zip(nemo_outputs, nest_outputs)):
            if isinstance(no, torch.Tensor) and isinstance(neo, torch.Tensor):
                print(f"\nOutput[{i}]:")
                print(f"  NeMo shape: {list(no.shape)}, dtype: {no.dtype}")
                print(f"  nest shape: {list(neo.shape)}, dtype: {neo.dtype}")
                
                if no.shape == neo.shape:
                    diff = (no.float() - neo.float()).abs()
                    print(f"  Max diff: {diff.max().item():.6e}")
                    print(f"  Mean diff: {diff.mean().item():.6e}")
                    
                    # 找到最大差异的位置
                    max_idx = diff.argmax().item()
                    unraveled = []
                    temp = max_idx
                    for dim in reversed(no.shape):
                        unraveled.insert(0, temp % dim)
                        temp //= dim
                    
                    print(f"  Max diff at index: {unraveled}")
                    print(f"    NeMo value: {no.flatten()[max_idx].item():.6f}")
                    print(f"    nest value: {neo.flatten()[max_idx].item():.6f}")
                    
                    # 检查是否有 NaN 或 Inf
                    if torch.isnan(no).any():
                        print(f"  [WARNING] NeMo output has NaN!")
                    if torch.isnan(neo).any():
                        print(f"  [WARNING] nest output has NaN!")
                    if torch.isinf(no).any():
                        print(f"  [WARNING] NeMo output has Inf!")
                    if torch.isinf(neo).any():
                        print(f"  [WARNING] nest output has Inf!")
                    
                    # 统计信息
                    print(f"\n  NeMo stats: min={no.min().item():.6f}, max={no.max().item():.6f}, mean={no.float().mean().item():.6f}")
                    print(f"  nest stats: min={neo.min().item():.6f}, max={neo.max().item():.6f}, mean={neo.float().mean().item():.6f}")
                    
                    # 检查输出长度（第二个输出通常是 seq_len）
                    if i == 1:  # seq_len
                        print(f"\n  NeMo seq_len values: {no.tolist()}")
                        print(f"  nest seq_len values: {neo.tolist()}")
                        if not torch.equal(no, neo):
                            print(f"  [CRITICAL] seq_len mismatch! This affects all downstream computations.")
                else:
                    print(f"  [MISMATCH] Shape mismatch!")
    
    # 检查 mel filterbank
    print("\n" + "="*80)
    print("检查 Mel Filterbank (fb buffer)")
    print("="*80)
    
    # 尝试从模型结构中获取 filterbank
    nemo_struct_path = nemo_step_dir.parent / 'model_structure.pt'
    nest_struct_path = nest_step_dir.parent / 'model_structure.pt'
    
    if nemo_struct_path.exists() and nest_struct_path.exists():
        nemo_struct = torch.load(nemo_struct_path, map_location='cpu', weights_only=False)
        nest_struct = torch.load(nest_struct_path, map_location='cpu', weights_only=False)
        
        # 查找 fb (filterbank) buffer
        nemo_fb = None
        nest_fb = None
        
        for name, param in nemo_struct.items():
            if 'fb' in name.lower() and 'featurizer' in name.lower():
                nemo_fb = param
                print(f"NeMo fb found: {name}, shape={param.shape}")
                break
        
        for name, param in nest_struct.items():
            if 'fb' in name.lower() and 'featurizer' in name.lower():
                nest_fb = param
                print(f"nest fb found: {name}, shape={param.shape}")
                break
        
        if nemo_fb is not None and nest_fb is not None:
            if nemo_fb.shape == nest_fb.shape:
                diff = (nemo_fb.float() - nest_fb.float()).abs()
                print(f"\nFilterbank comparison:")
                print(f"  Max diff: {diff.max().item():.6e}")
                print(f"  Mean diff: {diff.mean().item():.6e}")
                if diff.max().item() > 1e-6:
                    print(f"  [CRITICAL] Mel filterbank is different!")
                    print(f"  This is likely due to different librosa versions.")
                else:
                    print(f"  [OK] Mel filterbank matches")
            else:
                print(f"  [MISMATCH] Shape mismatch: {nemo_fb.shape} vs {nest_fb.shape}")
    
    # 检查 window buffer
    print("\n" + "="*80)
    print("检查 Window Function (window buffer)")
    print("="*80)
    
    if nemo_struct_path.exists() and nest_struct_path.exists():
        nemo_window = None
        nest_window = None
        
        for name, param in nemo_struct.items():
            if 'window' in name.lower() and 'featurizer' in name.lower():
                nemo_window = param
                print(f"NeMo window found: {name}, shape={param.shape}")
                break
        
        for name, param in nest_struct.items():
            if 'window' in name.lower() and 'featurizer' in name.lower():
                nest_window = param
                print(f"nest window found: {name}, shape={param.shape}")
                break
        
        if nemo_window is not None and nest_window is not None:
            if nemo_window.shape == nest_window.shape:
                diff = (nemo_window.float() - nest_window.float()).abs()
                print(f"\nWindow function comparison:")
                print(f"  Max diff: {diff.max().item():.6e}")
                if diff.max().item() > 1e-6:
                    print(f"  [WARNING] Window function is different!")
                else:
                    print(f"  [OK] Window function matches")


if __name__ == '__main__':
    main()

