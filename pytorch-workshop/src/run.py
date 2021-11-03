import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from .data import SimCLRDataset
from .model import SimCLR
from .trainer import Trainer

cifar10 = CIFAR10(root='./data', train=True, download=True)
images = cifar10.data[:100]
targets = cifar10.targets

device = 'cpu'
batch_size = 10
epochs = 2

train_images, val_images = train_test_split(images, test_size=0.2)
train_dataset = SimCLRDataset(train_images)
val_dataset = SimCLRDataset(val_images)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

simclr_model = SimCLR().to(device)

optimiser = optim.Adam(simclr_model.parameters())


trainer = Trainer(simclr_model, optimiser, device, batch_size, epochs)

trainer.train(train_loader, val_loader)
