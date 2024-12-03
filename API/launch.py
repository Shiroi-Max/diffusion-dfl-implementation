"""Launch module"""

from abc import abstractmethod
from typing import Callable, List, Optional

import PIL
import torch
from torchvision import transforms

from API.dfl_ddpm import DecentralizedFLDDPM, NodeInfo
from API.filtered_dataset import (
    FilteredDataset,
    FilteredEMNIST,
)


class Launch:
    def __init__(
        self,
        config,
        list_neighbours: Optional[List[List[int]]] = None,
        list_labels: Optional[List[List[int]]] = None,
    ):
        self.config = config
        self.list_labels = list_labels
        self.list_neighbours = list_neighbours

        self.train = self.list_neighbours is not None

    def _transform_image(self, image: PIL.Image.Image) -> torch.Tensor:
        preprocess_image = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
            ]
        )
        return preprocess_image(image).to(self.config.device)

    def _transform_image(self, target: int) -> torch.Tensor:
        return torch.tensor(target, device=self.config.device)

    @abstractmethod
    def init_data_set(self, path: str, index: int = -1) -> FilteredDataset:
        return FilteredDataset(
            path,
            base_class=self.config.dataset,
            train=self.config.train,
            download=True,
            threshold=self.config.threshold,
            labels=(
                self.list_labels[index]
                if index != -1
                and len(self.list_labels) > 0
                and len(self.list_labels[index]) > 0
                else None
            ),
            transform=self._transform_image,
            target_transform=self._transform_image,
        )

    def _init_node(self, index: int = -1):
        return NodeInfo(
            self.init_data_set(self.config.dataset_path, index).dataset,
            self.list_neighbours[index] if index != -1 else None,
        )

    def launch(self):
        if not self.train:
            return

        global_labels = self.init_data_set(self.config.dataset_path).labels
        nodes_info = [self._init_node(i) for i in range(len(self.list_neighbours))]

        dfl_ddpm = DecentralizedFLDDPM(self.config, global_labels, nodes_info)
        dfl_ddpm.run()


class EMNISTLaunch(Launch):
    def init_data_set(self, path: str, index: int = -1) -> FilteredDataset:
        return FilteredEMNIST(
            path,
            split=self.config.dataset.split("_")[1],
            train=self.config.train,
            download=True,
            threshold=self.config.threshold,
            labels=(
                self.list_labels[index]
                if index != -1
                and len(self.list_labels) > 0
                and len(self.list_labels[index]) > 0
                else None
            ),
            transform=self._transform_image,
            target_transform=self._transform_image,
        )

    def _transform_image(self, image: PIL.Image.Image) -> torch.Tensor:
        extra_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),
                transforms.RandomHorizontalFlip(p=1.0),
            ]
        )
        image = extra_transforms(image)
        return super()._transform_image(image)


def get_launch(dataset: str) -> Callable:
    complex_dataset_class_map = {
        "emnist": EMNISTLaunch,
    }

    launch_class = complex_dataset_class_map.get(dataset.split("_")[0])
    if not launch_class:
        return Launch
    return launch_class
