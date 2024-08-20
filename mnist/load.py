import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class MNIST:
    def __init__(self, data_path, img_size):
        self.data_path = data_path
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.train_dataset = torchvision.datasets.MNIST(
            root=data_path, train=True, download=True, transform=self.transform)
        self.test_dataset = torchvision.datasets.MNIST(
            root=data_path, download=True, transform=self.transform)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=64, shuffle=False)
