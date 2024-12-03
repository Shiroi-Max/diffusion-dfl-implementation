"""Module to filter the dataset based on the labels and threshold"""

import importlib
from typing import List, Optional

import torch
from torchvision.datasets import EMNIST


class FilteredDataset:
    def __init__(
        self,
        root: str,
        base_class: str,
        *args,
        threshold: int,
        labels: Optional[List[int]] = None,
        **kwargs,
    ):
        module = importlib.import_module("torchvision.datasets")
        base_class = getattr(module, base_class.upper())
        self.dataset = base_class(root, *args, **kwargs)
        self.labels = labels
        self.threshold = threshold
        self.adjust_labels = lambda: None
        self._filter_data()

    def _filter_data(self):
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
    def __init__(self, *args, **kwargs):
        self.adjust_labels = self._adjust_emnist_labels
        super().__init__(*args, base_class=EMNIST, **kwargs)

    def _adjust_emnist_labels(self):
        self.labels = self.labels[:-1]
        self.dataset.targets = self.dataset.targets - 1
