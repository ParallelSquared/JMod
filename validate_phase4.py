#!/usr/bin/env python
"""
Validation script for Phase 4 changes.
Runs a small test and saves output for comparison.
"""

import subprocess
import os
import hashlib
import json
import pandas as pd
import numpy as np

def run_test_and_save_output(output_dir, tag):
    """Run JMod with test config and save results."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run with test config
    print(f"Running test for {tag}...")
    cmd = [
        "python", "run_jmod.py", 
        "--config_json", "data/test_config_mzml.json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save stdout and stderr
    with open(f"{output_dir}/stdout_{tag}.txt", "w") as f:
        f.write(result.stdout)
    
    with open(f"{output_dir}/stderr_{tag}.txt", "w") as f:
        f.write(result.stderr)
    
    # Save return code
    with open(f"{output_dir}/returncode_{tag}.txt", "w") as f:
        f.write(str(result.returncode))
    
    print(f"Test completed with return code: {result.returncode}")
    
    # If successful, save output files
    if result.returncode == 0:
        output_folder = "./data/test_results_mzml/"
        if os.path.exists(output_folder):
            # Find the most recent results folder
            folders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]
            if folders:
                latest_folder = max(folders, key=lambda f: os.path.getmtime(os.path.join(output_folder, f)))
                results_path = os.path.join(output_folder, latest_folder)
                
                # Copy key output files
                for filename in ["decoylibsearch_coeffs.csv", "all_IDs.csv", "filtered_IDs.csv"]:
                    src = os.path.join(results_path, filename)
                    if os.path.exists(src):
                        dst = os.path.join(output_dir, f"{filename}_{tag}")
                        subprocess.run(["cp", src, dst])
                        print(f"Saved {filename}")
                        
                        # Calculate checksum
                        with open(src, 'rb') as f:
                            checksum = hashlib.md5(f.read()).hexdigest()
                        with open(f"{output_dir}/checksum_{filename}_{tag}.txt", "w") as f:
                            f.write(checksum)

def compare_outputs(dir1, dir2):
    """Compare outputs from two test runs."""
    
    print("\n=== Comparing Outputs ===")
    
    # Compare return codes
    with open(f"{dir1}/returncode_before.txt") as f:
        rc1 = f.read().strip()
    with open(f"{dir2}/returncode_after.txt") as f:
        rc2 = f.read().strip()
    
    print(f"Return codes: before={rc1}, after={rc2}")
    
    # Compare key output files
    for filename in ["decoylibsearch_coeffs.csv", "all_IDs.csv", "filtered_IDs.csv"]:
        file1 = f"{dir1}/{filename}_before"
        file2 = f"{dir2}/{filename}_after"
        
        if os.path.exists(file1) and os.path.exists(file2):
            # Compare checksums
            with open(f"{dir1}/checksum_{filename}_before.txt") as f:
                sum1 = f.read().strip()
            with open(f"{dir2}/checksum_{filename}_after.txt") as f:
                sum2 = f.read().strip()
            
            if sum1 == sum2:
                print(f"✓ {filename}: Identical (checksum match)")
            else:
                print(f"✗ {filename}: Different checksums")
                
                # Try to load and compare as CSV
                try:
                    df1 = pd.read_csv(file1)
                    df2 = pd.read_csv(file2)
                    
                    print(f"  Shape: {df1.shape} vs {df2.shape}")
                    
                    if df1.shape == df2.shape:
                        # Compare numeric columns
                        numeric_cols = df1.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if col in df2.columns:
                                max_diff = np.abs(df1[col] - df2[col]).max()
                                if max_diff > 1e-6:
                                    print(f"  Column '{col}': max difference = {max_diff}")
                except Exception as e:
                    print(f"  Could not compare as CSV: {e}")
        else:
            print(f"✗ {filename}: Missing file")

def main():
    """Main validation workflow."""
    
    # Step 1: Run test before changes
    print("=== Running test BEFORE Phase 4 changes ===")
    run_test_and_save_output("validation_output", "before")
    
    print("\n" + "="*50)
    print("Phase 4 changes can now be applied.")
    print("After making changes, run this script again with --after flag")
    print("="*50)

if __name__ == "__main__":
    import sys
    
    if "--after" in sys.argv:
        # Run test after changes and compare
        print("=== Running test AFTER Phase 4 changes ===")
        run_test_and_save_output("validation_output", "after")
        
        # Compare results
        compare_outputs("validation_output", "validation_output")
    else:
        main()