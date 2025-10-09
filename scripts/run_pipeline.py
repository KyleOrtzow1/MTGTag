"""Convenience script to run the complete MTGTag pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path
import logging

def run_command(command: list, description: str) -> bool:
    """
    Run a command and return success status.

    Args:
        command: List of command components
        description: Description of what the command does

    Returns:
        True if command succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed with exit code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Run the complete MTGTag pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete MTGTag training pipeline"
    )
    parser.add_argument(
        "--labeled-data",
        type=Path,
        default="data/labeled_subset.csv",
        help="Path to labeled subset CSV file"
    )
    parser.add_argument(
        "--skip-diagnosis",
        action="store_true",
        help="Skip tag diagnosis step"
    )
    parser.add_argument(
        "--skip-cleaning",
        action="store_true",
        help="Skip tag cleaning step"
    )
    parser.add_argument(
        "--skip-domain-adapt",
        action="store_true",
        help="Skip domain adaptation step"
    )

    args = parser.parse_args()

    # Pipeline steps
    steps = []

    if not args.skip_diagnosis:
        steps.append(([
            sys.executable, "-m", "mtgtag.pipeline.diagnose",
            str(args.labeled_data)
        ], "Tag Diagnosis"))

    if not args.skip_cleaning:
        steps.append(([
            sys.executable, "-m", "mtgtag.pipeline.clean",
            str(args.labeled_data),
            "data/labeled_subset_clean.csv"
        ], "Tag Cleaning"))

    if not args.skip_domain_adapt:
        steps.append(([
            sys.executable, "-m", "mtgtag.pipeline.domain_adapt"
        ], "Domain Adaptation"))

    steps.extend([
        ([sys.executable, "-m", "mtgtag.pipeline.train"], "Classifier Training"),
        ([sys.executable, "-m", "mtgtag.pipeline.optimize"], "Threshold Optimization"),
        ([sys.executable, "-m", "mtgtag.pipeline.classify"], "Bulk Classification")
    ])

    # Run pipeline
    print("Starting MTGTag Pipeline")
    print(f"Total steps: {len(steps)}")

    for i, (command, description) in enumerate(steps, 1):
        print(f"\n[Step {i}/{len(steps)}] {description}")

        if not run_command(command, description):
            print(f"\nPipeline failed at step {i}: {description}")
            sys.exit(1)

    print("\n" + "="*60)
    print("🎉 Pipeline completed successfully!")
    print("Check the output files for results.")
    print("="*60)

if __name__ == "__main__":
    main()