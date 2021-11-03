import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .nt_xent import NTXentLoss


class Trainer:

    def __init__(self, simclr_model, optimiser, device, batch_size, epochs):
        self.simclr_model = simclr_model
        self.optimiser = optimiser
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs

        self.loss_function = NTXentLoss(
            device, batch_size, 0.5, True
        )

    def compute_loss(self, xis, xjs):
        his, zis = self.simclr_model(xis)
        hjs, zjs = self.simclr_model(xjs)

        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.loss_function(zis, zjs)

        return loss

    def validate(self, val_loader):
        with torch.no_grad(): # disable autograd engine (no backprop)
            self.simclr_model.eval() # dropout, batchnorm in inference mode

            valid_loss = 0.0
            counter = 0
            for xis, xjs in val_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self.compute_loss(xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        self.simclr_model.train()

        return valid_loss

    def train(self, train_loader, val_loader):
        best_valid_loss = np.inf

        for epoch in range(self.epochs):
            print('Epoch {} of {}'.format(epoch, self.epochs))

            for xis, xjs in train_loader:
                self.optimiser.zero_grad()

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self.compute_loss(xis, xjs)
                print(loss.item())

                loss.backward() # compute gradients

                self.optimiser.step() # Update params based on gradients

            valid_loss = self.validate(val_loader)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                torch.save(
                    self.simclr_model.state_dict(),
                    'model.pth'
                )
