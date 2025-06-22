"""
Module to filter torchvision datasets based on label inclusion and class-wise sampling thresholds.

This is useful for simulating non-IID (non-independent and identically distributed) data distributions in federated learning environments.

Includes support for standard datasets (e.g., MNIST, FashionMNIST) and EMNIST with label adjustments.
"""

import importlib
from typing import List, Optional

import torch
from torchvision.datasets import EMNIST


class FilteredDataset:
    """
    A wrapper for torchvision datasets that filters the data based on a given list of labels
    and a sampling threshold for non-selected classes.

    Attributes
    ----------
    dataset : torchvision.dataset
        The underlying filtered dataset.
    labels : List[int]
        The list of labels included in the dataset.
    threshold : int
        Percentage of samples to include for excluded classes.
    adjust_labels : Callable
        Optional function for modifying labels (overridden by subclasses if needed).
    """

    def __init__(
        self,
        root: str,
        base_class: str,
        *args,
        threshold: int,
        labels: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        root : str
            Root directory of the dataset.
        base_class : str
            Name of the torchvision dataset class (e.g., 'MNIST', 'FashionMNIST').
        threshold : int
            Percentage of samples to retain from excluded classes.
        labels : Optional[List[int]]
            List of class labels to fully include.
        """
        module = importlib.import_module("torchvision.datasets")
        base_class = getattr(module, base_class.upper())
        self.dataset = base_class(root, *args, **kwargs)
        self.labels = labels
        self.threshold = threshold
        self.adjust_labels = lambda: None
        self._filter_data()

    def _filter_data(self):
        """
        Filters the dataset to include only specified labels, and includes a percentage
        of other labels based on the given threshold.
        """
        if self.labels is not None:
            included_mask = (
                sum((self.dataset.targets == label) for label in self.labels) > 0
            )
            all_classes = set(range(len(self.dataset.classes)))
            excluded_classes = all_classes - set(self.labels)

            for label in excluded_classes:
                label_mask = self.dataset.targets == label
                num_samples_to_include = int(sum(label_mask) * self.threshold / 100)

                if num_samples_to_include > 0:
                    indices = torch.where(label_mask)[0]
                    selected_indices = torch.randperm(len(indices))[
                        :num_samples_to_include
                    ]
                    included_mask[indices[selected_indices]] = True

            self.dataset.data = self.dataset.data[included_mask]
            self.dataset.targets = self.dataset.targets[included_mask]
        else:
            self.labels = list(range(len(self.dataset.classes)))

        if self.adjust_labels is not None and callable(self.adjust_labels):
            self.adjust_labels()


class FilteredEMNIST(FilteredDataset):
    """
    EMNIST-specific filtered dataset wrapper that adjusts label range from [1, 26] to [0, 25].
    Useful for the 'letters' split which uses label 1-based indexing.
    """

    def __init__(self, *args, **kwargs):
        self.adjust_labels = self._adjust_emnist_labels
        super().__init__(*args, base_class=EMNIST, **kwargs)

    def _adjust_emnist_labels(self):
        """
        Adjust EMNIST labels to start from 0 instead of 1, which is required for
        compatibility with the classifier.
        """
        self.labels = self.labels[:-1]  # remove label 26
        self.dataset.targets = self.dataset.targets - 1
