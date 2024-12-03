import sys

sys.path.insert(0, "/home/maxim/diffusers-venv/diffusion-implementation")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from API.utils import SimpleCNN

import torch
import torch.nn as nn
import torch.optim as optim


class AdjustedEMNIST(Dataset):
    def __init__(self, emnist_dataset):
        self.emnist_dataset = emnist_dataset

    def __getitem__(self, index):
        img, label = self.emnist_dataset[index]
        # Ajustar las etiquetas para que estén en el rango [0, 25]
        label -= 1
        return img, label

    def __len__(self):
        return len(self.emnist_dataset)


# Transformaciones para las imágenes EMNIST Letters
transform = transforms.Compose(
    [
        transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),
        transforms.Lambda(lambda img: transforms.functional.hflip(img)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     (0.5,), (0.5,)
        # ),  # Normalizar con la media y desviación estándar de EMNIST Letters
    ]
)

# Cargar el conjunto de datos EMNIST Letters
train_dataset = datasets.EMNIST(
    root="./laboratory/datasets",
    split="letters",
    train=True,
    transform=transform,
    download=False,
)
test_dataset = datasets.EMNIST(
    root="./laboratory/datasets",
    split="letters",
    train=False,
    transform=transform,
    download=False,
)

# Ajustar los conjuntos de datos
adjusted_train_dataset = AdjustedEMNIST(train_dataset)
adjusted_test_dataset = AdjustedEMNIST(test_dataset)


train_loader = DataLoader(adjusted_train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(adjusted_test_dataset, batch_size=64, shuffle=False)


# Crear el modelo
model = SimpleCNN(26).to("cuda")

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
num_epochs = 10
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

# Evaluar el modelo
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

# Guardar el modelo
torch.save(model.state_dict(), "emnist_letters-classifier.pth")
