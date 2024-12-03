"""Main script to run the training of a model on a given dataset and topology"""

import os
import time

from API.launch import get_launch
from API.utils import (
    TrainingConfig,
    extract_labels,
    extract_neighbours,
    format_time,
    parse_args,
)


def get_config(
    dataset: str, topology: str, threshold: int = 0, overwrite_output_dir: bool = False
):
    return TrainingConfig(
        topology=topology,
        dataset=dataset,
        threshold=threshold,
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


def main(config: TrainingConfig):
    topology_dir = os.path.join(
        os.path.dirname(__file__),
        "laboratory",
        "topologies",
        f"topology-{config.topology}",
    )
    if not os.path.exists(topology_dir):
        raise ValueError("Topology does not exist")

    neighbours_txt = os.path.join(topology_dir, "neighbours.txt")
    labels_txt = os.path.join(topology_dir, f"labels-{config.dataset}.txt")

    launch_class = get_launch(config.dataset)
    launch_class(
        config, extract_neighbours(neighbours_txt), extract_labels(labels_txt)
    ).launch()


if __name__ == "__main__":
    start = time.time()
    main(get_config(*parse_args()))
    end = time.time()
    print("Time taken: ", format_time(end - start), " seconds")
