#!/usr/bin/env python3
"""Simple training script for experiments."""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Simple training script")
    parser.add_argument("--config-path", type=str, help="Configuration path")
    parser.add_argument("--config-name", type=str, help="Configuration name")

    args, unknown = parser.parse_known_args()

    # Parse additional hydra-style arguments
    overrides = {}
    for arg in unknown:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                overrides[key] = int(value)
            except ValueError:
                overrides[key] = value

    print(f"Training with config: {args.config_path}/{args.config_name}")
    if overrides:
        print(f"Overrides: {overrides}")

    # For now, just simulate training
    print("Starting simulated training...")
    print("Training completed successfully!")

    return 0

if __name__ == "__main__":
    sys.exit(main())