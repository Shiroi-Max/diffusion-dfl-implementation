from torchvision import transforms
from abc import abstractmethod
from typing import List, Optional, Callable
from API.filtered_dataset import (
    FilteredDataset,
    FilteredMNIST,
    FilteredFashionMNIST,
    FilteredEMNIST,
)
from API.dfl_ddpm import run, NodeInfo

import torch, PIL


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

    def transformImage(self, image: PIL.Image.Image):
        preprocessImage = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
            ]
        )
        return preprocessImage(image).to(self.config.device)

    def transformTarget(self, target: int):
        return torch.tensor(target, device="cuda")

    @abstractmethod
    def initDataSet(self, path: str, index: int = -1) -> FilteredDataset:
        pass

    def initNode(self, index: int = -1):
        return NodeInfo(
            self.initDataSet(self.config.dataset_path, index).dataset,
            self.list_neighbours[index] if index != -1 else None,
        )

    def launch(self):
        if not self.train:
            return

        global_labels = self.initDataSet(self.config.dataset_path).labels
        nodes_info = [self.initNode(i) for i in range(len(self.list_neighbours))]

        run(self.config, global_labels, nodes_info)


class MNISTLaunch(Launch):
    def initDataSet(self, path: str, index: int = -1) -> FilteredDataset:
        return FilteredMNIST(
            path,
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
            transform=self.transformImage,
            target_transform=self.transformTarget,
        )


class FashionMNISTLaunch(Launch):
    def initDataSet(self, path: str, index: int = -1) -> FilteredDataset:
        return FilteredFashionMNIST(
            path,
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
            transform=self.transformImage,
            target_transform=self.transformTarget,
        )


class EMNISTLaunch(Launch):
    def initDataSet(self, path: str, index: int = -1) -> FilteredDataset:
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
            transform=self.transformImage,
            target_transform=self.transformTarget,
        )

    def transformImage(self, image: PIL.Image.Image):
        extra_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),
                transforms.Lambda(lambda img: transforms.functional.hflip(img)),
            ]
        )
        image = extra_transforms(image)
        return super().transformImage(image)


def getLaunch(dataset: str) -> Callable:
    dataset_class_map = {
        "mnist": MNISTLaunch,
        "emnist": EMNISTLaunch,
        "fashionmnist": FashionMNISTLaunch,
    }

    launch_class = dataset_class_map.get(dataset)
    if not launch_class:
        raise ValueError(f"Dataset {dataset} not supported")
    return launch_class
