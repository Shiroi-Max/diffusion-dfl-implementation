"""
Entrypoint for executing training or evaluation of the DFL Diffusion pipeline.

This script parses CLI arguments to determine whether to run training or testing,
loads the appropriate YAML configuration, and delegates execution to the corresponding routine.
Execution time is measured and displayed at the end.

Usage (from CLI):
    python run.py train mnist ring
    python run.py test emnist ring --split letters
"""

import time

from src.core.cli import parse_args
from src.core.config import load_test_config_from_yaml, load_training_config_from_yaml
from src.core.filesystem import format_time
from src.core.testing import run_evaluation
from src.core.training import run_training


def run():
    """
    Main function for executing either training or testing based on CLI arguments.

    It measures the total execution time and reports it at the end. The configuration
    is dynamically loaded based on the specified mode and dataset/topology.

    Raises
    ------
    ValueError
        If the specified mode is not supported.
    """
    mode, config_path = parse_args()

    start = time.time()

    if mode == "train":
        config = load_training_config_from_yaml(config_path)
        run_training(config)
    elif mode == "test":
        config = load_test_config_from_yaml(config_path)
        run_evaluation(config)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    end = time.time()
    print("✅ Finished in:", format_time(end - start))


if __name__ == "__main__":
    run()
