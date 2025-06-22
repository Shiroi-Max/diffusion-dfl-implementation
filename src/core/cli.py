"""
Command-line argument parser for the DFL Diffusion training and evaluation pipeline.

This module handles user-provided arguments to configure execution of either training
or testing modes. It determines the appropriate YAML configuration file path based on
dataset, topology, and execution mode (train/test), ensuring consistency across
experiments.

Raises
------
FileNotFoundError
    If the expected configuration file does not exist in the `configs/` directory.
ValueError
    If the provided EMNIST split is not among the supported options.
"""

import argparse
import os


def parse_args() -> tuple[str, str]:
    """
    Parses CLI arguments and returns the selected mode and YAML config path.

    Returns
    -------
    tuple[str, str]
        A tuple containing the execution mode ("train" or "test") and the path to the YAML config file.

    Raises
    ------
    ValueError
        If the dataset is 'emnist' and no split is provided.
        If the split is invalid or used with a non-emnist dataset.
        If the configuration file does not exist.
    """
    parser = argparse.ArgumentParser(
        description="Run training or evaluation for the DFL Diffusion pipeline."
    )

    parser.add_argument("mode", choices=["train", "test"], help="Execution mode")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., mnist, emnist)")
    parser.add_argument("topology", type=str, help="Topology name (e.g., ring, custom)")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split of the dataset (only allowed for EMNIST: 'mnist', 'balanced', 'letters', 'digits')",
    )

    args = parser.parse_args()

    dataset = args.dataset.lower()
    topology = args.topology.lower()
    mode = args.mode.lower()

    # Validate split usage
    if dataset == "emnist":
        if not args.split:
            parser.error(
                "The argument '--split' is required when using the 'emnist' dataset."
            )
        if args.split not in ["mnist", "balanced", "letters", "digits"]:
            raise ValueError(f"Invalid EMNIST split: {args.split}")
        dataset += f"_{args.split}"
    elif args.split:
        parser.error(
            "The argument '--split' is only valid when using the 'emnist' dataset."
        )

    # Construct config filename
    filename = f"{mode}_{dataset}.yaml"
    config_path = os.path.join("configs", topology, filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return mode, config_path
