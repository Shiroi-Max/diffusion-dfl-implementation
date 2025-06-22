"""
Evaluation pipeline for testing generated images from trained diffusion models.

This module handles the process of evaluating class-conditional diffusion models
by generating synthetic images and comparing the predicted labels against the
expected ones using a pretrained classifier.

It uses TorchMetrics for accuracy computation and assumes models are stored
in `config.input_dir`, and logs are saved to `config.output_dir`.

Functions
---------
- run_evaluation(config): Main entrypoint for running evaluation on all node models.
- evaluate_node(...): Runs inference and evaluation for a single model directory.
- evaluate(...): Computes accuracy and writes results to a log file.
"""

import os

import torch
import torchmetrics
from torchvision.transforms import Compose, Resize, ToTensor

from src.core.config import TestConfig
from src.core.filesystem import reset
from src.core.launch import get_launch
from src.core.pipeline import DDPMConditionalPipeline
from src.data.classifier import SimpleCNN, get_classifier


def evaluate(
    pred: torch.Tensor,
    true: torch.Tensor,
    num_classes: int,
    log_path: str,
):
    """
    Compute classification accuracy and log results.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted class labels.
    true : torch.Tensor
        Ground-truth class labels.
    num_classes : int
        Total number of classes.
    log_path : str
        File path to store the evaluation log.
    """
    acc = torchmetrics.functional.accuracy(
        pred, true, task="multiclass", average="macro", num_classes=num_classes
    )
    result = f"Accuracy: {acc * 100:.2f}%"
    print(result)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(result + "\n")


def evaluate_node(
    model_dir: str,
    labels: list[int],
    samples: int,
    classifier: torch.nn.Module,
    transform,
    config: TestConfig,
):
    """
    Run evaluation on a single diffusion model directory.

    Parameters
    ----------
    model_dir : str
        Path to the saved model directory.
    labels : list[int]
        List of label IDs used for generation.
    samples : int
        Number of samples per label.
    classifier : torch.nn.Module
        Pretrained classifier for inference.
    transform : torchvision.transforms.Compose
        Transformations to apply to generated images.
    config : TestConfig
        Test configuration object.
    """
    print(f"Testing model {os.path.basename(model_dir)}")

    pipeline = DDPMConditionalPipeline.from_pretrained(model_dir).to(config.device)

    true = (
        torch.tensor([[label] * samples for label in labels])
        .flatten()
        .to(config.device)
    )
    images = torch.stack(
        [transform(img) for img in pipeline(true, true.size(0)).images]
    ).to(config.device)

    output = classifier(images)
    _, predicted = torch.max(output.data, 1)

    log_file = f"node{os.path.basename(model_dir)[-1] if os.path.basename(model_dir)[-1] != '0' else '10'}.txt"
    evaluate(predicted, true, len(labels), os.path.join(config.output_dir, log_file))


def run_evaluation(config: TestConfig):
    """
    Run evaluation across all models under the given test configuration.

    Parameters
    ----------
    config : TestConfig
        Configuration containing dataset, model, and evaluation paths.
    """
    launch_class = get_launch(config.dataset)
    dataset_class = launch_class(config).init_data_set(config.dataset_path)
    labels = dataset_class.labels
    samples = config.samples // len(labels)

    if samples == 0 or config.samples > len(dataset_class.dataset):
        raise ValueError("Invalid number of samples for evaluation")

    if config.overwrite_output_dir and os.path.isdir(config.output_dir):
        reset(config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    if not os.path.exists(config.classifier_path):
        get_classifier(config.dataset)(
            config.classifier_path, config.dataset, config.device
        ).launch()

    classifier = SimpleCNN(len(labels))
    classifier.load_state_dict(torch.load(config.classifier_path))
    classifier.to(config.device)

    transform = Compose([Resize((28, 28)), ToTensor()])

    for model_dir in os.scandir(config.input_dir):
        if os.path.isdir(model_dir):
            evaluate_node(
                model_dir.path, labels, samples, classifier, transform, config
            )
