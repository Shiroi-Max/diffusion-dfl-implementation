"""This module contains utility functions and classes for the API"""

import shutil
import os
import argparse

from typing import List, Tuple
from dataclasses import dataclass

import torch

from PIL.Image import Image
from diffusers.utils import make_image_grid

from API.pipeline import DDPMConditionalPipeline


@dataclass
class TrainingConfig:
    topology: str  # the topology to use
    dataset: str  # the dataset to use
    threshold: int  # the threshold of samples to use for each label not specified
    image_size: int  # the generated image resolution
    train_batch_size: int  # the batch size for training
    num_epochs: int  # the number of epochs to train
    gradient_accumulation_steps: int  # the number of steps to accumulate gradients
    learning_rate: int  # the learning rate for the optimizer
    lr_warmup_steps: int  # number of warmup steps for the learning rate scheduler
    save_epochs: int  # save model and sample in x epochs
    mixed_precision: str  # `no` for float32, `fp16` for automatic mixed precision
    num_processes: int  # the number of processes to use for training
    device: str  # the device to use for training
    overwrite_output_dir: bool  # whether to overwrite the output directory
    train: bool = True  # whether to train the model

    @property  # the root directory of the run
    def root_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "..", "laboratory")

    @property  # the directory of the dataset
    def output_dir(self) -> str:
        return os.path.join(
            self.root_dir,
            "scenarios",
            f"topology-{self.topology}",
            f"{self.dataset}-{self.image_size}-{self.threshold}%",
        )

    @property  # the directory to save the dataset
    def dataset_path(self) -> str:
        return os.path.join(self.root_dir, "datasets")


@dataclass
class TestConfig:
    topology: str  # the topology to use
    dataset: str  # the dataset to use
    threshold: int  # the threshold of samples to use for each label not specified
    image_size: int  # the generated image resolution
    samples: int  # the number of samples to generate for evaluation
    device: str  # the device to use for testing
    overwrite_output_dir: bool  # whether to overwrite the output directory
    train: bool = False  # whether to train the model

    @property  # the root directory of the run
    def root_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), "..", "laboratory")

    @property  # the directory of the models
    def input_dir(self) -> str:
        return os.path.join(
            self.root_dir,
            "scenarios",
            f"topology-{self.topology}",
            f"{self.dataset}-{self.image_size}-{self.threshold}%",
            "models",
        )

    @property  # the directory to save the results
    def output_dir(self) -> str:
        return os.path.join(
            self.root_dir,
            "evaluations",
            f"topology-{self.topology}",
            f"{self.dataset}-{self.image_size}-{self.threshold}%",
        )

    @property  # the directory of the classifier
    def classifier_path(self) -> str:
        return os.path.join(
            self.root_dir, "classifiers", f"{self.dataset}-classifier.pth"
        )

    @property  # the directory of the dataset
    def dataset_path(self) -> str:
        return os.path.join(self.root_dir, "datasets")


def read_matrix_from_file(file_path: str) -> List[List[int]]:
    matrix = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            row = [int(num) for num in line.strip().split()]
            matrix.append(row)
    return matrix


def extract_neighbours(file_path: str) -> List[List[bool]]:
    matrix = read_matrix_from_file(file_path)
    neighbours = []
    for i, row in enumerate(matrix):
        neighbour = [False] * len(matrix)
        neighbour[i] = True
        for j in row:
            neighbour[j - 1] = True
        neighbours.append(neighbour)

    return neighbours


def extract_labels(file_path: str) -> List[List[int]]:
    return read_matrix_from_file(file_path)


def reset(path: str):
    for filename in os.scandir(path):
        try:
            if os.path.isfile(filename) or os.path.islink(filename):
                os.unlink(filename)
            elif os.path.isdir(filename):
                print(f"Deleting directory {filename}")
                shutil.rmtree(filename)
        except OSError as e:
            print(f"Failed to delete {filename}. Reason: {e}")
    os.rmdir(path)


def generate(
    root: str,
    labels: List[int],
    device: str,
    epoch: int,
    index: int,
    pipeline: DDPMConditionalPipeline,
):
    y = torch.tensor([label for label in labels]).flatten().to(device)
    images = pipeline(y, y.size(0)).images

    grid = make_image_grid(images, len(labels), 1)

    sample(
        root,
        grid,
        epoch + 1,
        index,
        str(labels[0]) + "-" + str(labels[-1]),
    )


def sample(root: str, grid: Image, epoch: int, index: int, name: str):
    os.makedirs(root, exist_ok=True)
    model_dir = os.path.join(root, f"model{index}")
    os.makedirs(model_dir, exist_ok=True)
    epoch_dir = os.path.join(model_dir, f"{epoch:02d}")
    os.makedirs(epoch_dir, exist_ok=True)

    grid.save(os.path.join(epoch_dir, name + "-sample.png"))


def parse_args() -> Tuple[str, str, int, bool]:
    parser = argparse.ArgumentParser(
        description="Process arguments to execute the model"
    )

    parser.add_argument("dataset", type=str, help="Name of the dataset.")
    parser.add_argument("topology", type=int, help="Network topology to use.")

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Split of the dataset (e.g., 'letters').",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Threshold for the shared labels strategy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Indicates wether the previous results should be overwritten.",
    )

    args = parser.parse_args()

    if args.dataset.lower() == "emnist" and not args.split:
        parser.error(
            "The argument '--split' is compulsory when the 'emnist' dataset is used."
        )

    if args.split and args.split not in ["mnist", "balanced", "letters", "digits"]:
        raise ValueError("Invalid emnist split")

    dataset = args.dataset.lower()

    if dataset not in ["mnist", "fashionmnist", "emnist"]:
        raise ValueError("Invalid dataset")

    if args.split:
        dataset += "_" + args.split

    return dataset, args.topology, args.threshold, args.overwrite


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    if minutes > 0:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"
