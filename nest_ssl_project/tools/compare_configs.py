#!/usr/bin/env python3
"""
Compare NeMo and nest_ssl_project configurations to ensure they match.
"""

import sys
import os
import yaml
from pathlib import Path

def load_yaml(file_path):
    """Load YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def compare_dicts(d1, d2, path="", differences=None):
    """Recursively compare two dictionaries."""
    if differences is None:
        differences = []
    
    # Get all keys from both dicts
    all_keys = set(d1.keys()) | set(d2.keys())
    
    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key
        
        if key not in d1:
            differences.append(f"Missing in nest_ssl: {current_path}")
            continue
        if key not in d2:
            differences.append(f"Missing in NeMo: {current_path}")
            continue
        
        val1 = d1[key]
        val2 = d2[key]
        
        # Skip _target_ differences (we use different import paths)
        if key == "_target_":
            continue
        
        if isinstance(val1, dict) and isinstance(val2, dict):
            compare_dicts(val1, val2, current_path, differences)
        elif isinstance(val1, list) and isinstance(val2, list):
            if val1 != val2:
                differences.append(f"Different values at {current_path}: nest_ssl={val1}, NeMo={val2}")
        else:
            if val1 != val2:
                # Special handling for string interpolation
                if isinstance(val1, str) and isinstance(val2, str):
                    if val1.startswith("${") and val2.startswith("${"):
                        # Both are interpolations, check if they reference the same path
                        if val1 != val2:
                            differences.append(f"Different interpolation at {current_path}: nest_ssl={val1}, NeMo={val2}")
                    elif val1 != val2:
                        differences.append(f"Different values at {current_path}: nest_ssl={val1}, NeMo={val2}")
                else:
                    differences.append(f"Different values at {current_path}: nest_ssl={val1}, NeMo={val2}")
    
    return differences

def main():
    # Get paths
    project_root = Path(__file__).parent.parent
    nemo_root = project_root.parent / "NeMo"
    
    nest_config_path = project_root / "config" / "nest_fast-conformer.yaml"
    nemo_config_path = nemo_root / "examples" / "asr" / "conf" / "ssl" / "nest" / "nest_fast-conformer.yaml"
    
    print("="*80)
    print("Comparing NeMo and nest_ssl_project Configurations")
    print("="*80)
    print(f"\nNeMo config: {nemo_config_path}")
    print(f"nest_ssl config: {nest_config_path}")
    
    if not nest_config_path.exists():
        print(f"\nERROR: nest_ssl config not found at {nest_config_path}")
        sys.exit(1)
    
    if not nemo_config_path.exists():
        print(f"\nERROR: NeMo config not found at {nemo_config_path}")
        print("Please ensure NeMo is cloned in the parent directory.")
        sys.exit(1)
    
    # Load configs
    print("\nLoading configurations...")
    nest_config = load_yaml(nest_config_path)
    nemo_config = load_yaml(nemo_config_path)
    
    # Compare
    print("\nComparing configurations...")
    differences = compare_dicts(nest_config, nemo_config)
    
    # Print results
    print("\n" + "="*80)
    if not differences:
        print("✓ Configurations are identical (ignoring _target_ paths)")
    else:
        print(f"⚠ Found {len(differences)} differences:")
        print("="*80)
        for diff in differences:
            print(f"  - {diff}")
    
    # Check critical model parameters
    print("\n" + "="*80)
    print("Critical Model Parameters Comparison")
    print("="*80)
    
    critical_params = [
        ("model.num_classes", "num_classes"),
        ("model.num_books", "num_books"),
        ("model.code_dim", "code_dim"),
        ("model.mask_position", "mask_position"),
        ("model.encoder.n_layers", "n_layers"),
        ("model.encoder.d_model", "d_model"),
        ("model.encoder.n_heads", "n_heads"),
        ("model.encoder.conv_kernel_size", "conv_kernel_size"),
        ("model.encoder.subsampling_factor", "subsampling_factor"),
        ("model.encoder.subsampling_conv_channels", "subsampling_conv_channels"),
        ("model.encoder.use_bias", "use_bias"),
        ("model.encoder.xscaling", "xscaling"),
        ("model.optim.lr", "lr"),
        ("model.optim.weight_decay", "weight_decay"),
        ("model.optim.betas", "betas"),
    ]
    
    def get_nested_value(config, path):
        """Get nested value from config."""
        keys = path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    all_match = True
    for path, name in critical_params:
        nest_val = get_nested_value(nest_config, path)
        nemo_val = get_nested_value(nemo_config, path)
        
        if nest_val == nemo_val:
            print(f"✓ {name}: {nest_val}")
        else:
            print(f"✗ {name}: nest_ssl={nest_val}, NeMo={nemo_val}")
            all_match = False
    
    print("\n" + "="*80)
    if all_match:
        print("✓ All critical parameters match!")
    else:
        print("⚠ Some critical parameters differ!")
    print("="*80)
    
    return 0 if all_match and not differences else 1

if __name__ == "__main__":
    sys.exit(main())

