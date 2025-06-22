"""
Configuration module for training and evaluation in the DFL Diffusion pipeline.

This module defines the data classes for training and testing configurations,
including computed properties for managing input/output paths in the project
workspace. It also provides helper functions to load configuration settings
from YAML files.

Classes
-------
- TrainingConfig : Holds parameters and paths for model training.
- TestConfig     : Holds parameters and paths for model evaluation.

Functions
---------
- load_training_config_from_yaml(path): Loads a TrainingConfig from a YAML file.
- load_test_config_from_yaml(path): Loads a TestConfig from a YAML file.
"""

import os
from dataclasses import dataclass

import yaml


@dataclass
class TrainingConfig:
    """
    Configuration structure for training a model in the DFL Diffusion pipeline.

    Attributes
    ----------
    topology : str
        Name of the network topology used (e.g., "ring", "custom").
    dataset : str
        Name of the dataset to use (e.g., "mnist", "emnist_letters").
    threshold : int
        Minimum number of samples per label per node.
    image_size : int
        Size of generated images (resolution).
    train_batch_size : int
        Batch size for training.
    num_epochs : int
        Number of training epochs.
    gradient_accumulation_steps : int
        Gradient accumulation steps before optimizer update.
    learning_rate : float
        Learning rate for the optimizer.
    lr_warmup_steps : int
        Number of steps for learning rate warm-up.
    save_epochs : int
        Frequency of saving model checkpoints.
    mixed_precision : str
        Mixed precision mode: "no" or "fp16".
    num_processes : int
        Number of participating nodes (processes).
    device : str
        Device used for training: "cuda" or "cpu".
    overwrite_output_dir : bool
        Whether to overwrite the previous output directory.
    train : bool
        Whether to enable training mode (always True for this class).
    """

    topology: str
    dataset: str
    threshold: int
    image_size: int
    train_batch_size: int
    num_epochs: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_warmup_steps: int
    save_epochs: int
    mixed_precision: str
    num_processes: int
    device: str
    overwrite_output_dir: bool
    train: bool = True

    @property
    def root_dir(self) -> str:
        """Root path of the laboratory workspace."""
        return os.path.join(os.path.dirname(__file__), "..", "..", "laboratory")

    @property
    def input_dir(self) -> str:
        """Path to the folder containing the topology configuration."""
        return os.path.join(self.root_dir, "topologies", self.topology)

    @property
    def output_dir(self) -> str:
        """Directory where training outputs and checkpoints are stored."""
        return os.path.join(
            self.root_dir,
            "scenarios",
            self.topology,
            f"{self.dataset}-{self.image_size}-{self.threshold}%",
        )

    @property
    def dataset_path(self) -> str:
        """Path to the dataset directory."""
        return os.path.join(self.root_dir, "datasets")


@dataclass
class TestConfig:
    """
    Configuration structure for evaluating a trained model in the DFL Diffusion pipeline.

    Attributes
    ----------
    topology : str
        Name of the network topology used (e.g., "ring", "custom").
    dataset : str
        Name of the dataset to use.
    threshold : int
        Minimum number of samples per label per node.
    image_size : int
        Size of generated images.
    samples : int
        Number of samples to generate per evaluation.
    device : str
        Device used for evaluation.
    overwrite_output_dir : bool
        Whether to overwrite previous evaluation outputs.
    train : bool
        Always False for testing.
    """

    topology: str
    dataset: str
    threshold: int
    image_size: int
    samples: int
    device: str
    overwrite_output_dir: bool
    train: bool = False

    @property
    def root_dir(self) -> str:
        """Root path of the laboratory workspace."""
        return os.path.join(os.path.dirname(__file__), "..", "..", "laboratory")

    @property
    def input_dir(self) -> str:
        """Path to the folder containing the trained models."""
        return os.path.join(
            self.root_dir,
            "scenarios",
            self.topology,
            f"{self.dataset}-{self.image_size}-{self.threshold}%",
            "models",
        )

    @property
    def output_dir(self) -> str:
        """Directory where evaluation results are stored."""
        return os.path.join(
            self.root_dir,
            "evaluations",
            self.topology,
            f"{self.dataset}-{self.image_size}-{self.threshold}%",
        )

    @property
    def classifier_path(self) -> str:
        """Path to the auxiliary classifier used for accuracy evaluation."""
        return os.path.join(
            self.root_dir, "classifiers", f"{self.dataset}-classifier.pth"
        )

    @property
    def dataset_path(self) -> str:
        """Path to the dataset directory."""
        return os.path.join(self.root_dir, "datasets")


def load_training_config_from_yaml(path: str) -> TrainingConfig:
    """
    Load a training configuration from a YAML file.

    Parameters
    ----------
    path : str
        File path to the training YAML configuration.

    Returns
    -------
    TrainingConfig
        An initialized training configuration object.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    return TrainingConfig(**raw_config)


def load_test_config_from_yaml(path: str) -> TestConfig:
    """
    Load a testing configuration from a YAML file.

    Parameters
    ----------
    path : str
        File path to the testing YAML configuration.

    Returns
    -------
    TestConfig
        An initialized testing configuration object.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    return TestConfig(**raw_config)
