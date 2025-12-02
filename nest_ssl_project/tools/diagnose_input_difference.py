#!/usr/bin/env python3
"""
诊断输入数据差异的工具脚本。

这个脚本会：
1. 加载保存的 batch 数据
2. 比较每个字段的详细差异
3. 显示数据统计信息
4. 帮助找出输入不一致的原因
"""

import sys
from pathlib import Path
import torch
import numpy as np

def load_batch_data(batch_path):
    """加载 batch 数据，处理各种格式。"""
    import sys
    import importlib
    
    # Add project paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    parent_root = project_root.parent
    
    for path in [str(project_root), str(parent_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Try to import modules for pickle
    try:
        sys.modules['data'] = importlib.import_module('nest_ssl_project.data')
        sys.modules['data.ssl_dataset'] = importlib.import_module('nest_ssl_project.data.ssl_dataset')
    except ModuleNotFoundError:
        try:
            sys.modules['data'] = importlib.import_module('data')
            sys.modules['data.ssl_dataset'] = importlib.import_module('data.ssl_dataset')
        except ModuleNotFoundError:
            pass
    
    batch = torch.load(batch_path, map_location='cpu', weights_only=False)
    return batch

def extract_tensors(batch_obj):
    """提取所有 tensor 字段。"""
    tensors = {}
    
    if isinstance(batch_obj, dict):
        for k, v in batch_obj.items():
            if isinstance(v, torch.Tensor):
                tensors[k] = v
    elif hasattr(batch_obj, '__dict__') or hasattr(type(batch_obj), '__annotations__'):
        for attr in ['audio', 'audio_len', 'noise', 'noise_len', 'noisy_audio', 'noisy_audio_len', 'sample_id']:
            if hasattr(batch_obj, attr):
                val = getattr(batch_obj, attr)
                if isinstance(val, torch.Tensor):
                    tensors[attr] = val
        if hasattr(batch_obj, '__dict__'):
            for k, v in batch_obj.__dict__.items():
                if isinstance(v, torch.Tensor) and k not in tensors:
                    tensors[k] = v
    
    return tensors

def compare_tensors(t1, t2, name, atol=1e-5, rtol=1e-5):
    """比较两个 tensor。"""
    if t1.shape != t2.shape:
        return {
            'match': False,
            'reason': f'Shape mismatch: {t1.shape} vs {t2.shape}',
            'max_diff': None,
            'mean_diff': None,
        }
    
    if t1.dtype != t2.dtype:
        t1 = t1.to(t2.dtype)
    
    # Use float tensors for numeric stats if original dtype is not floating point
    if torch.is_floating_point(t1):
        t1_float = t1
        t2_float = t2
    else:
        t1_float = t1.to(torch.float32)
        t2_float = t2.to(torch.float32)
    
    diff = (t1_float - t2_float).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    is_close = torch.allclose(t1_float, t2_float, atol=atol, rtol=rtol)
    
    return {
        'match': is_close,
        'reason': 'Match' if is_close else f'Max diff: {max_diff:.2e}',
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        't1_stats': {
            'min': t1_float.min().item(),
            'max': t1_float.max().item(),
            'mean': t1_float.mean().item(),
            'std': t1_float.std().item(),
        },
        't2_stats': {
            'min': t2_float.min().item(),
            'max': t2_float.max().item(),
            'mean': t2_float.mean().item(),
            'std': t2_float.std().item(),
        },
    }

def main():
    nemo_batch_path = Path('saved_nemo_outputs/step_1/batch.pt')
    nest_batch_path = Path('saved_nest_outputs/step_1/batch.pt')
    
    if not nemo_batch_path.exists():
        print(f"Error: NeMo batch file not found: {nemo_batch_path}")
        return
    
    if not nest_batch_path.exists():
        print(f"Error: nest batch file not found: {nest_batch_path}")
        return
    
    print("Loading batch data...")
    nemo_batch = load_batch_data(nemo_batch_path)
    nest_batch = load_batch_data(nest_batch_path)
    
    print("\nExtracting tensors...")
    nemo_tensors = extract_tensors(nemo_batch)
    nest_tensors = extract_tensors(nest_batch)
    
    print(f"\nNeMo batch fields: {sorted(nemo_tensors.keys())}")
    print(f"nest batch fields: {sorted(nest_tensors.keys())}")
    
    common_keys = set(nemo_tensors.keys()) & set(nest_tensors.keys())
    nemo_only = set(nemo_tensors.keys()) - set(nest_tensors.keys())
    nest_only = set(nest_tensors.keys()) - set(nemo_tensors.keys())
    
    if nemo_only:
        print(f"\n⚠️  NeMo only fields: {sorted(nemo_only)}")
    if nest_only:
        print(f"\n⚠️  nest only fields: {sorted(nest_only)}")
    
    print(f"\n{'='*80}")
    print("Field-by-field comparison:")
    print(f"{'='*80}")
    
    for key in sorted(common_keys):
        nemo_val = nemo_tensors[key]
        nest_val = nest_tensors[key]
        
        print(f"\n{key}:")
        print(f"  NeMo: shape={list(nemo_val.shape)}, dtype={nemo_val.dtype}")
        print(f"  nest: shape={list(nest_val.shape)}, dtype={nest_val.dtype}")
        
        comp = compare_tensors(nemo_val, nest_val, key)
        
        if comp['match']:
            print(f"  ✓ Match")
        else:
            print(f"  ✗ Mismatch: {comp['reason']}")
            if comp['max_diff'] is not None:
                print(f"    Max diff: {comp['max_diff']:.2e}")
                print(f"    Mean diff: {comp['mean_diff']:.2e}")
            
            print(f"\n  NeMo stats:")
            print(f"    min={comp['t1_stats']['min']:.6f}, max={comp['t1_stats']['max']:.6f}")
            print(f"    mean={comp['t1_stats']['mean']:.6f}, std={comp['t1_stats']['std']:.6f}")
            
            print(f"\n  nest stats:")
            print(f"    min={comp['t2_stats']['min']:.6f}, max={comp['t2_stats']['max']:.6f}")
            print(f"    mean={comp['t2_stats']['mean']:.6f}, std={comp['t2_stats']['std']:.6f}")
            
            # Show sample values
            if nemo_val.numel() > 0:
                print(f"\n  NeMo sample (first 10): {nemo_val.flatten()[:10].tolist()}")
                print(f"  nest sample (first 10): {nest_val.flatten()[:10].tolist()}")
                
                # Check if values are completely different or just slightly different
                if comp['max_diff'] > 1.0:
                    print(f"\n  ⚠️  Large difference detected! Values are significantly different.")
                    print(f"      This suggests different data samples or preprocessing.")
                elif comp['max_diff'] > 1e-3:
                    print(f"\n  ⚠️  Moderate difference detected. Values are similar but not identical.")
                    print(f"      This suggests numerical precision differences or minor preprocessing differences.")
                else:
                    print(f"\n  ℹ️  Small difference detected. Values are very similar.")
                    print(f"      This might be acceptable numerical precision differences.")

if __name__ == '__main__':
    main()

