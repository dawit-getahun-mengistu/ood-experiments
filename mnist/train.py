import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Define a simple CNN model


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    './data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training function


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing function


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# OOD detection function


def ood_detection(model, device, test_loader, ood_loader):
    model.eval()
    in_distribution_scores = []
    out_of_distribution_scores = []

    with torch.no_grad():
        # In-distribution data
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            scores = torch.max(output, dim=1)[0]
            in_distribution_scores.extend(scores.cpu().numpy())

        # Out-of-distribution data
        for data, _ in ood_loader:
            data = data.to(device)
            output = model(data)
            scores = torch.max(output, dim=1)[0]
            out_of_distribution_scores.extend(scores.cpu().numpy())

    # Calculate metrics
    in_distribution_scores = np.array(in_distribution_scores)
    out_of_distribution_scores = np.array(out_of_distribution_scores)

    threshold = np.percentile(in_distribution_scores, 80)
    in_distribution_detection = (in_distribution_scores >= threshold).mean()
    out_of_distribution_detection = (
        out_of_distribution_scores < threshold).mean()

    print(f"In-distribution detection rate: {in_distribution_detection:.2f}")
    print(
        f"Out-of-distribution detection rate: {out_of_distribution_detection:.2f}")


# Main execution
if __name__ == '__main__':
    # Train the model
    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, epoch)

    # Test the model
    print(f"Test set result for MNIST")
    test(model, device, test_loader)

    # Load an OOD dataset (e.g., Fashion-MNIST)
    ood_dataset = datasets.FashionMNIST(
        './data', train=False, download=True, transform=transform)
    ood_loader = DataLoader(ood_dataset, batch_size=1000, shuffle=False)

    print(f"Test set result for fashion-MNIST")
    test(model, device, ood_loader)

    # Perform OOD detection
    ood_detection(model, device, test_loader, ood_loader)
