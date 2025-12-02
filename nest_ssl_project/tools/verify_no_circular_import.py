#!/usr/bin/env python3
"""
Verify that training_output_saver.py has no circular imports.
"""

import sys
from pathlib import Path

def check_file(file_path):
    """Check if file has circular import."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    issues = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Check for self-import
        if 'from tools.training_output_saver import' in stripped and 'TrainingOutputSaver' in stripped:
            if not stripped.startswith('#'):  # Not a comment
                issues.append(f"Line {i}: Found circular import: {stripped}")
    
    return issues

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    saver_file = script_dir / 'training_output_saver.py'
    
    if not saver_file.exists():
        print(f"ERROR: File not found: {saver_file}")
        sys.exit(1)
    
    print(f"Checking {saver_file}...")
    issues = check_file(saver_file)
    
    if issues:
        print("ERROR: Found circular import issues:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease remove the circular import statement.")
        sys.exit(1)
    else:
        print("OK: No circular imports found.")
        
        # Try to import
        try:
            sys.path.insert(0, str(script_dir.parent))
            from tools.training_output_saver import TrainingOutputSaver
            print("OK: Import successful.")
        except ImportError as e:
            print(f"ERROR: Import failed: {e}")
            sys.exit(1)

