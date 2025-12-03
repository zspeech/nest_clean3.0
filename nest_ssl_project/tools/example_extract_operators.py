#!/usr/bin/env python3
"""
Example script showing how to extract PyTorch operators from a model.
This can be integrated into your training script.
"""

import torch
from pathlib import Path
import json

# Import the extraction function
from nest_ssl_project.tools.get_model_operators import extract_aten_operators, save_operators_json


def extract_operators_from_training_output(model, dummy_input, output_dir: Path):
    """
    Extract operators from a model and save to JSON.
    
    Args:
        model: PyTorch model instance
        dummy_input: Dummy input tensor
        output_dir: Directory to save the JSON file
    """
    print("Extracting ATen operators from model...")
    
    # Extract operators
    operators = extract_aten_operators(model, dummy_input, verbose=True)
    
    if not operators:
        print("Warning: No operators extracted. Model tracing may have failed.")
        return
    
    # Prepare model info
    model_info = {
        "input_shape": list(dummy_input.shape),
        "model_type": type(model).__name__,
    }
    
    # Save to JSON
    output_path = output_dir / "pytorch_operators.json"
    save_operators_json(operators, output_path, model_info=model_info)
    
    return operators


# Example usage in training script:
if __name__ == '__main__':
    print("""
Example usage in your training script:

    from nest_ssl_project.tools.get_model_operators import extract_aten_operators, save_operators_json
    import torch
    from pathlib import Path
    
    # After model initialization
    asr_model = EncDecDenoiseMaskedTokenPredModel(cfg=cfg.model, trainer=trainer)
    
    # Create dummy input matching your model's expected input
    dummy_input = torch.randn(2, 80, 1584)  # [B, D, T] for preprocessor output
    
    # Extract operators
    operators = extract_aten_operators(asr_model, dummy_input, verbose=True)
    
    # Save to JSON
    output_dir = Path("./saved_outputs/step_0")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_operators_json(operators, output_dir / "pytorch_operators.json",
                       model_info={"model_type": "EncDecDenoiseMaskedTokenPredModel"})
    """)

