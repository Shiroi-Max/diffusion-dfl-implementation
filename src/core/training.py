"""
Training runner module for decentralized diffusion models.

This module orchestrates the initialization and execution of the training process
for Denoising Diffusion Probabilistic Models (DDPMs) in a Decentralized Federated
Learning (DFL) context.

It loads topology metadata from YAML files, selects the appropriate launch strategy
(e.g., standard or EMNIST-specific), and executes the training loop using the
provided `TrainingConfig`.

Functions
---------
- run_training(config): Runs the training process for a given configuration.
"""

import os

from src.core.config import TrainingConfig
from src.core.launch import get_launch
from src.core.loaders_yaml import extract_labels_yaml, extract_neighbours_yaml


def run_training(config: TrainingConfig):
    """
    Run the decentralized training process based on the given configuration.

    This function:
    - Validates that the topology directory exists
    - Loads neighbour and label configurations from YAML files
    - Initializes the correct Launch subclass based on dataset
    - Triggers the training routine using the selected launch class

    Parameters
    ----------
    config : TrainingConfig
        The training configuration containing paths, parameters, and topology info.

    Raises
    ------
    ValueError
        If the topology directory does not exist or is misconfigured.
    """
    if not os.path.exists(config.input_dir):
        raise ValueError(f"Topology directory not found: {config.input_dir}")

    neighbours = extract_neighbours_yaml(
        os.path.join(config.input_dir, "neighbours.yaml")
    )
    labels = extract_labels_yaml(
        os.path.join(config.input_dir, f"labels-{config.dataset}.yaml")
    )

    launch_class = get_launch(config.dataset)
    launch_class(config, neighbours, labels).launch()
