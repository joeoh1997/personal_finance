import torch.nn as nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        resnet = models.resnet18(pretrained=False)

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        h = self.encoder(x).view(-1, 512)
        return h


class SimCLR(nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()

        self.encoder = Encoder() # f

        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, x):
        h = self.encoder(x)

        z = self.projection_head(h)

        return h, z
