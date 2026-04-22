# =============================================================================
# MAIN.PY - Execute All Phases Sequentially
# =============================================================================
# This script orchestrates the execution of all three phases:
#   Phase 1: Setup, Data Loading & Initial Inspection
#   Phase 2: Data Cleaning, EDA & Visualisation
#   Phase 3: Feature Engineering & Data Preparation
# =============================================================================

import subprocess
import sys
import os

# Project root directory
PROJECT_DIR = r'C:\Users\uzwal\OneDrive\Desktop\DS\Climate changes'

# Subdirectories
SCRIPTS_DIR = os.path.join(PROJECT_DIR, 'scripts')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'outputs')

os.chdir(PROJECT_DIR)

def run_phase(script_name, phase_name):
    """Run a Python script and handle errors."""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    
    print(f"\n{'=' * 60}")
    print(f"STARTING: {phase_name}")
    print(f"{'=' * 60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_DIR,
            check=True
        )
        print(f"\n{'=' * 60}")
        print(f"COMPLETED: {phase_name}")
        print(f"{'=' * 60}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {phase_name} failed with exit code {e.returncode}")
        return False

def main():
    print("=" * 60)
    print("CLIMATE CHANGE DATA PIPELINE")
    print("Executing Phase 1 → Phase 2 → Phase 3")
    print("=" * 60)
    
    # Phase 1: Setup and Loading
    if not run_phase("Phase1_Setup_and_Loading (1).py", "Phase 1: Setup & Loading"):
        sys.exit(1)
    
    # Phase 2: Cleaning and EDA
    if not run_phase("Phase2_Cleaning_and_EDA.py", "Phase 2: Cleaning & EDA"):
        sys.exit(1)
    
    # Phase 3: Feature Engineering
    if not run_phase("Phase3_Feature_Engineering (2).py", "Phase 3: Feature Engineering"):
        sys.exit(1)
    
    print("=" * 60)
    print("ALL PHASES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput files generated in {DATA_DIR}:")
    print("  - global_warming_clean.csv (from Phase 2)")
    print("  - global_warming_features.csv (from Phase 3)")
    print(f"\nGraphs saved to: {OUTPUTS_DIR}")

if __name__ == "__main__":
    main()