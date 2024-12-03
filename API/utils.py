from typing import List
from dataclasses import dataclass

import os
import torch.nn as nn


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


def readMatrixFromFile(file_path: str) -> List[List[int]]:
    matrix = []
    with open(file_path, "r") as file:
        for line in file:
            row = [int(num) for num in line.strip().split()]
            matrix.append(row)
    return matrix


def extractNeighbours(file_path: str) -> List[List[bool]]:
    matrix = readMatrixFromFile(file_path)
    neighbours = []
    for i in range(len(matrix)):
        neighbour = [False] * len(matrix)
        neighbour[i] = True
        for j in range(len(matrix[i])):
            neighbour[matrix[i][j] - 1] = True
        neighbours.append(neighbour)

    return neighbours


def extractLabels(file_path: str) -> List[List[int]]:
    return readMatrixFromFile(file_path)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
