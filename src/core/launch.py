"""
Launch module for initializing and running decentralized training or evaluation.

This module defines a launcher abstraction that handles dataset instantiation,
transformation, and the orchestration of the training process in a decentralized
federated learning setting. It supports specialization for specific datasets like EMNIST.

Classes
-------
- Launch: Base class for training/evaluation orchestration.
- EMNISTLaunch: Dataset-specific subclass for EMNIST preprocessing.
- get_launch: Factory function to select the appropriate launch class.
"""

from abc import abstractmethod
from typing import Callable, List, Optional

import PIL
import torch
from torchvision import transforms

from src.core.dfl_ddpm import DecentralizedFLDDPM, NodeInfo
from src.data.filtered_dataset import FilteredDataset, FilteredEMNIST


class Launch:
    """
    Base class to launch decentralized training or testing.

    This class handles:
    - Dataset initialization (one node or all nodes)
    - Image and label transformations
    - Launching training via the DecentralizedFLDDPM class

    Parameters
    ----------
    config : TrainingConfig or TestConfig
        Configuration object containing experiment parameters.
    list_neighbours : Optional[List[List[int]]]
        Adjacency list for each node's neighbours.
    list_labels : Optional[List[List[int]]]
        List of label indices assigned to each node.
    """

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
        """
        Preprocesses the input image by resizing and converting it to a tensor.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to be transformed.

        Returns
        -------
        torch.Tensor
            Transformed image tensor.
        """
        preprocess_image = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
            ]
        )
        return preprocess_image(image).to(self.config.device)

    def _transform_image(self, target: int) -> torch.Tensor:
        """
        Converts the label into a tensor.

        Parameters
        ----------
        target : int
            Label to be converted.

        Returns
        -------
        torch.Tensor
            Label as a tensor.
        """
        return torch.tensor(target, device=self.config.device)

    @abstractmethod
    def init_data_set(self, path: str, index: int = -1) -> FilteredDataset:
        """
        Initializes a dataset for training or evaluation.

        Parameters
        ----------
        path : str
            Path to the dataset directory.
        index : int, optional
            Node index. If -1, initializes the full dataset.

        Returns
        -------
        FilteredDataset
            A dataset with optional label filtering.
        """
        return FilteredDataset(
            path,
            base_class=self.config.dataset,
            train=self.config.train,
            download=True,
            threshold=self.config.threshold,
            labels=(
                self.list_labels[index]
                if index != -1 and self.list_labels and self.list_labels[index]
                else None
            ),
            transform=self._transform_image,
            target_transform=self._transform_image,
        )

    def _init_node(self, index: int = -1):
        """
        Builds a NodeInfo instance for a given index.

        Parameters
        ----------
        index : int
            Index of the node.

        Returns
        -------
        NodeInfo
            Contains the dataset and neighbours for the node.
        """
        return NodeInfo(
            self.init_data_set(self.config.dataset_path, index).dataset,
            self.list_neighbours[index] if index != -1 else None,
        )

    def launch(self):
        """
        Launches decentralized training if in training mode.

        Initializes all node datasets and starts the training loop using DecentralizedFLDDPM.
        """
        if not self.train:
            return

        global_labels = self.init_data_set(self.config.dataset_path).labels
        nodes_info = [self._init_node(i) for i in range(len(self.list_neighbours))]

        dfl_ddpm = DecentralizedFLDDPM(self.config, global_labels, nodes_info)
        dfl_ddpm.run()


class EMNISTLaunch(Launch):
    """
    Specialization of Launch for the EMNIST dataset.

    Applies specific image transformations such as rotation and flipping.
    """

    def init_data_set(self, path: str, index: int = -1) -> FilteredDataset:
        """
        Initializes the EMNIST dataset with appropriate split and transformations.

        Parameters
        ----------
        path : str
            Path to the dataset directory.
        index : int, optional
            Node index. If -1, initializes the full dataset.

        Returns
        -------
        FilteredEMNIST
            EMNIST dataset object with filtering and preprocessing.
        """
        return FilteredEMNIST(
            path,
            split=self.config.dataset.split("_")[1],
            train=self.config.train,
            download=True,
            threshold=self.config.threshold,
            labels=(
                self.list_labels[index]
                if index != -1 and self.list_labels and self.list_labels[index]
                else None
            ),
            transform=self._transform_image,
            target_transform=self._transform_image,
        )

    def _transform_image(self, image: PIL.Image.Image) -> torch.Tensor:
        """
        Applies EMNIST-specific preprocessing:
        - Rotates image -90 degrees
        - Applies horizontal flip

        Parameters
        ----------
        image : PIL.Image.Image
            Input image.

        Returns
        -------
        torch.Tensor
            Transformed image tensor.
        """
        extra_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),
                transforms.RandomHorizontalFlip(p=1.0),
            ]
        )
        image = extra_transforms(image)
        return super()._transform_image(image)


def get_launch(dataset: str) -> Callable:
    """
    Returns the appropriate Launch subclass based on the dataset.

    Parameters
    ----------
    dataset : str
        Name of the dataset (e.g., "mnist", "emnist_letters").

    Returns
    -------
    Callable
        Launch or EMNISTLaunch class depending on the dataset.
    """
    complex_dataset_class_map = {
        "emnist": EMNISTLaunch,
    }

    launch_class = complex_dataset_class_map.get(dataset.split("_")[0])
    if not launch_class:
        return Launch
    return launch_class
