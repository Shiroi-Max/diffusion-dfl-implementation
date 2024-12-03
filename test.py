from API.launch import getLaunch
from API.pipeline import DDPMConditionalPipeline
from API.utils import TestConfig, SimpleCNN
from torchvision.transforms import ToTensor, Resize, Compose
from typing import Tuple

import torch, sys, os, shutil
import torchmetrics


def get_config(
    dataset: str,
    topology: str,
    threshold: int,
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


def validate_args(argv) -> Tuple[str, str, int, bool]:
    if len(argv) < 4 or len(argv) > 6:
        raise ValueError(
            "Usage: python test.py <dataset> <topology> <threshold> [overwrite]"
        )

    dataset = argv[1]
    topology = None
    threshold = None
    overwrite = False

    if dataset not in ["mnist", "fashionmnist", "emnist"]:
        raise ValueError("Invalid dataset")

    if dataset == "emnist":
        if len(argv) < 5:
            raise ValueError(
                "Usage: python test.py emnist <emnist_split> <topology> <threshold> [overwrite]"
            )
        emnist_split = argv[2]
        if emnist_split not in ["mnist", "balanced", "letters", "digits"]:
            raise ValueError("Invalid emnist split")
        dataset += "_" + emnist_split
        topology = argv[3]
        threshold = int(argv[4])
        if len(argv) == 6 and argv[5] == "true":
            overwrite = True
    else:
        topology = argv[2]
        threshold = int(argv[3])
        if len(argv) == 5 and argv[4] == "true":
            overwrite = True

    return dataset, topology, threshold, overwrite


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

    f = open(log, "w")
    f.write(result_p + "\n")
    f.close()


def reset(path: str):
    for filename in os.scandir(path):
        try:
            if os.path.isfile(filename) or os.path.islink(filename):
                os.unlink(filename)
            elif os.path.isdir(filename):
                print("Deleting directory %s" % filename)
                shutil.rmtree(filename)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (filename, e))
    os.rmdir(path)


def main(dataset: str, topology: str, threshold: int, overwrite: bool):
    config = get_config(dataset, topology, threshold, overwrite)

    if not os.path.isdir(config.input_dir):
        raise ValueError("Input directory does not exist")

    launch_class = getLaunch(sys.argv[1])
    dataset_class = launch_class(config).initDataSet(config.dataset_path)
    labels = dataset_class.labels
    dataset = dataset_class.dataset
    samples_per_label = config.samples // len(labels)

    if samples_per_label == 0:
        raise ValueError("Not enough samples per label")
    if config.samples > len(dataset):
        raise ValueError("Not enough samples in the dataset")

    if config.overwrite_output_dir and os.path.isdir(config.output_dir):
        reset(config.output_dir)

    os.makedirs(config.output_dir, exist_ok=True)

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

        classifier = SimpleCNN(len(labels))
        classifier.load_state_dict(torch.load(config.classifier_path))
        classifier.to(config.device)

        len(generated_images)
        outputs = classifier(generated_images)
        _, predicted_labels = torch.max(outputs.data, 1)

        log = os.path.join(
            config.output_dir,
            f"node{filename.name[-1] if filename.name[-1] != '0' else '10'}.txt",
        )
        evaluate(predicted_labels, true_labels, len(labels), log)


if __name__ == "__main__":
    dataset, topology, threshold, overwrite = validate_args(sys.argv)
    main(dataset, topology, threshold, overwrite)
