import torch
import torch.nn as nn
import torch.nn.functional as F


class _2LayerCNN(nn.Module):

    def __init__(self, input_channels=3):
        super(_2LayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


class _3LayerCNN(nn.Module):

    def __init__(self, input_channels=3):
        super(_3LayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        # self.fc1 = nn.Linear(128 * 4 * 4, 128)

        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


class VGGInspiredCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(VGGInspiredCNN, self).__init__()
        # Convolutional block 1
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # Convolutional block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        # Convolutional block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Max pooling layer with a 2x2 kernel (applied after each conv block)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Assuming input size of 32x32, after three poolings: 32 -> 16 -> 8 -> 4.
        # The flattened feature dimension is then 256 * 4 * 4 = 4096.
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # first dense layer
        self.fc2 = nn.Linear(512, 128)          # second dense layer
        self.fc3 = nn.Linear(128, num_classes)  # decision layer

    def forward(self, x):
        # Convolutional Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten the feature maps for the fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Softmax activation for classification
        x = F.softmax(x, dim=1)
        return x
