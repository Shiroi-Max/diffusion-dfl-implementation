"""Utilities for training and testing a simple CNN classifier on the dataset specified."""

import os
from typing import Callable

import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import EMNIST, MNIST, FashionMNIST
from torchvision import transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier:
    def __init__(self, root: str, dataset: str, device: str):
        self.root = root
        self.transform = self._transform_image
        self.dataset = dataset
        self.device = device
        self.num_labels = None

    def _transform_image(self, image: PIL.Image.Image) -> torch.Tensor:
        return transforms.ToTensor()(image).to(self.device)

    def _get_dataset(self) -> Callable:
        dataset_class_map = {
            "mnist": MNIST,
            "emnist": EMNIST,
            "fashionmnist": FashionMNIST,
        }

        launch_class = dataset_class_map.get(self.dataset)
        if not launch_class:
            raise ValueError(f"Dataset {self.dataset} not supported")
        return launch_class

    def _prepare_data(self, dataset: str) -> tuple:
        dataset_root = os.path.join(
            os.path.dirname(__file__), "..", "laboratory", "datasets"
        )

        dataset = self._get_dataset()
        train_dataset = dataset(
            root=dataset_root,
            train=True,
            transform=self.transform,
            download=False,
        )
        test_dataset = dataset(
            root=dataset_root,
            train=False,
            transform=self.transform,
            download=False,
        )

        self.num_labels = len(train_dataset.classes)

        return train_dataset, test_dataset

    def _prepare_data_loader(self, train_dataset, test_dataset) -> tuple:
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        return train_loader, test_loader

    def _train(self, model, train_loader, criterion, optimizer, num_epochs):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to("cuda"), labels.to("cuda")

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    def _evaluate(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to("cuda"), labels.to("cuda")
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}%")

    def launch(self):
        train_dataset, test_dataset = self._prepare_data("fashionmnist")
        train_loader, test_loader = self._prepare_data_loader(
            train_dataset, test_dataset
        )

        model = SimpleCNN(self.num_labels).to("cuda")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 10
        self._train(model, train_loader, criterion, optimizer, num_epochs)
        self._evaluate(model, test_loader)

        torch.save(model.state_dict(), self.root)


class AdjustedEMNIST(Dataset):
    def __init__(self, emnist_dataset):
        self.emnist_dataset = emnist_dataset

    def __getitem__(self, index):
        img, label = self.emnist_dataset[index]
        # Adjust the labels so they are in the range [0, 25]
        label -= 1
        return img, label

    def __len__(self):
        return len(self.emnist_dataset)


class ClassifierEMNIST(Classifier):
    def _prepare_data(self, dataset: str) -> tuple:
        dataset_root = os.path.join(
            os.path.dirname(__file__), "..", "laboratory", "datasets"
        )

        dataset = self._get_dataset()
        train_dataset = dataset(
            root=dataset_root,
            train=True,
            split="letters",
            transform=self.transform,
            download=False,
        )
        test_dataset = dataset(
            root=dataset_root,
            train=False,
            split="letters",
            transform=self.transform,
            download=False,
        )
        # Adjust the datasets
        adjusted_train_dataset = AdjustedEMNIST(train_dataset)
        adjusted_test_dataset = AdjustedEMNIST(test_dataset)

        self.num_labels = len(train_dataset.classes) - 1

        return adjusted_train_dataset, adjusted_test_dataset


def get_classifier(dataset: str) -> Callable:
    complex_dataset_class_map = {
        "emnist": ClassifierEMNIST,
    }

    classifier_class = complex_dataset_class_map.get(dataset.split("_")[0])
    if not classifier_class:
        return Classifier
    return classifier_class
