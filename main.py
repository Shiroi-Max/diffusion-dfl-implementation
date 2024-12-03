from API.utils import extractNeighbours, extractLabels, TrainingConfig
from API.launch import getLaunch
from typing import Tuple

import sys, os, time


def get_config(dataset: str, topology: str, overwrite_output_dir: bool = False):
    return TrainingConfig(
        topology=topology,
        dataset=dataset,
        threshold=20,
        image_size=32,
        train_batch_size=128,
        num_epochs=100,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_warmup_steps=500,
        save_epochs=10,
        mixed_precision="fp16",
        num_processes=2,
        device="cuda",
        overwrite_output_dir=overwrite_output_dir,
    )


def validate_args(argv) -> Tuple[str, str, bool]:
    if len(argv) < 3 or len(argv) > 5:
        raise ValueError("Usage: python main.py <dataset> <topology> [overwrite]")

    dataset = argv[1]
    topology = None
    overwrite = False

    if dataset not in ["mnist", "fashionmnist", "emnist"]:
        raise ValueError("Invalid dataset")

    if dataset == "emnist":
        if len(argv) < 4:
            raise ValueError(
                "Usage: python main.py emnist <emnist_split> <topology> [overwrite]"
            )
        emnist_split = argv[2]
        if emnist_split not in ["mnist", "balanced", "letters", "digits"]:
            raise ValueError("Invalid emnist split")
        dataset += "_" + emnist_split
        topology = argv[3]
        if len(argv) == 5 and argv[4] == "true":
            overwrite = True
    else:
        topology = argv[2]
        if len(argv) == 4 and argv[3] == "true":
            overwrite = True

    return dataset, topology, overwrite


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    elif minutes > 0:
        return f"{minutes}m {remaining_seconds}s"
    else:
        return f"{remaining_seconds}s"


def main(dataset: str, topology: str, overwrite: bool):
    config = get_config(dataset, topology, overwrite)
    topology_dir = os.path.join(
        os.path.dirname(__file__),
        "laboratory",
        "topologies",
        f"topology-{topology}",
    )
    if not os.path.exists(topology_dir):
        raise ValueError("Topology does not exist")

    neighbours_txt = os.path.join(topology_dir, "neighbours.txt")
    labels_txt = os.path.join(topology_dir, f"labels-{dataset}.txt")

    launch_class = getLaunch(sys.argv[1])
    launch_class(
        config, extractNeighbours(neighbours_txt), extractLabels(labels_txt)
    ).launch()


if __name__ == "__main__":
    dataset, topology, overwrite = validate_args(sys.argv)
    start = time.time()
    main(dataset, topology, overwrite)
    end = time.time()
    print("Time taken: ", format_time(end - start), " seconds")
