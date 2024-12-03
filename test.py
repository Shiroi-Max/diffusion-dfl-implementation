"""Test the model on the dataset and save the results in the output directory"""

import os
import sys

import torch
import torchmetrics
from torchvision.transforms import Compose, Resize, ToTensor

from API.launch import get_launch
from API.pipeline import DDPMConditionalPipeline
from API.utils import TestConfig, reset, parse_args
from API.classifier import SimpleCNN, get_classifier


def get_config(
    dataset: str,
    topology: str,
    threshold: int = 0,
    overwrite_output_dir: bool = False,
):
    return TestConfig(
        topology=topology,
        dataset=dataset,
        threshold=threshold,
        image_size=32,
        samples=500,
        device="cuda",
        overwrite_output_dir=overwrite_output_dir,
    )


transform = Compose([Resize((28, 28)), ToTensor()])


def evaluate(
    predicted_labels: torch.Tensor,
    true_labels: torch.Tensor,
    num_classes: int,
    log: str,
):
    # Calculate the Precision
    accuracy = torchmetrics.functional.accuracy(
        predicted_labels,
        true_labels,
        task="multiclass",
        average="macro",
        num_classes=num_classes,
    )
    result_p = f"Accuracy: {accuracy * 100}"
    print(result_p)

    with open(log, "w", encoding="utf-8") as f:
        f.write(result_p + "\n")


def main(config: TestConfig):
    if not os.path.isdir(config.input_dir):
        raise ValueError("Input directory does not exist")

    launch_class = get_launch(config.dataset)
    dataset_class = launch_class(config).init_data_set(config.dataset_path)
    labels = dataset_class.labels
    samples_per_label = config.samples // len(labels)

    if samples_per_label == 0:
        raise ValueError("Not enough samples per label")
    if config.samples > len(dataset_class.dataset):
        raise ValueError("Not enough samples in the dataset")

    if config.overwrite_output_dir and os.path.isdir(config.output_dir):
        reset(config.output_dir)

    os.makedirs(config.output_dir, exist_ok=True)

    if not os.path.exists(config.classifier_path):
        classifier_class = get_classifier(config.dataset)
        classifier_class(config.classifier_path, config.dataset, config.device).launch()

    classifier = SimpleCNN(len(labels))
    classifier.load_state_dict(torch.load(config.classifier_path))
    classifier.to(config.device)

    for filename in os.scandir(config.input_dir):
        if not os.path.isdir(filename):
            raise ValueError(f"{filename} should be a directory")

        print(f"Testing model {filename.name}")
        model_path = os.path.join(config.input_dir, filename.name)
        pipeline = DDPMConditionalPipeline.from_pretrained(model_path).to(config.device)

        true_labels = (
            torch.tensor([[labels[i]] * samples_per_label for i in range(len(labels))])
            .flatten()
            .to(config.device)
        )
        generated_images = pipeline(true_labels, true_labels.size(0)).images

        generated_images = torch.stack(
            [transform(image) for image in generated_images]
        ).to(config.device)

        len(generated_images)
        outputs = classifier(generated_images)
        _, predicted_labels = torch.max(outputs.data, 1)

        log = os.path.join(
            config.output_dir,
            f"node{filename.name[-1] if filename.name[-1] != '0' else '10'}.txt",
        )
        evaluate(predicted_labels, true_labels, len(labels), log)


if __name__ == "__main__":
    main(get_config(*parse_args()))
