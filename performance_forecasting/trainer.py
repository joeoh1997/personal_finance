# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:22:43 2021

@author: joeoh
"""

import os
import pickle
import random
from typing import Sequence

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as functional
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence

from matplotlib import pyplot as plt



import pandas as pd

from performance_forecasting.model import revenueForecastLSTM
#from performance_forecasting.torch_mod_functions import pad_sequence


OPTIMIZERS = {
    'adam':  optim.Adam,
    'rms_prop': optim.RMSprop,
    'adam_grad': optim.Adagrad,
    'adam_sparse': optim.SparseAdam,
    'asgd': optim.ASGD
}

LOSS_FUNCTIONS = {
    'l1': functional.l1_loss,
    'l2': functional.mse_loss
}

class Trainer:

    def __init__(
        self,
        sequence_pkl_path, 
        variables_to_forecast,
        selected_variables,
        layer_sizes,
        weight_decays,
        lr=0.0001,
        batch_size=32,
        optim='adam_grad',
        loss_function='l1',
        activation=None,
        use_bn=True,
        use_rnn=False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_indexes = [selected_variables.index(var) for var in variables_to_forecast]
        self.num_features = len(selected_variables)

        self.lstm = revenueForecastLSTM(
            self.num_features, 
            len(variables_to_forecast), 
            layer_sizes,
            use_bn=use_bn,
            use_rnn=use_rnn,
            dropout=weight_decays['lstm_dropout']
        ).to(self.device)

        global_params, fc_params = [], []

        for mod_name, module in self.lstm.named_modules():
            if mod_name == '':
                pass

            elif mod_name == 'lstm':
                global_params.extend(module.parameters())

            else:
                for param in module.parameters():
                    if len(param.shape) == 1:
                        global_params.append(param)
                    else:
                        fc_params.append(param)

        self.optimizer = OPTIMIZERS[optim](
            [
                {'params': global_params, 'weight_decay': weight_decays['global_l2']},
                {'params': fc_params, 'weight_decay': weight_decays['fc_l2']}
            ], lr=lr,
        )
        #self.lstm.parameters()
        self.loss_function = LOSS_FUNCTIONS[loss_function]

        self.train = []
        self.test = []
        
        self.init_h0 , self.init_c0 = [
            torch.zeros(
                layer_sizes['num_stacked_lstm_cells'],
                batch_size,
                layer_sizes['lstm_hidden_size']
            ).to(self.device)
            for _ in range(2)
        ]

        self.batch_size = batch_size
        self.activation = activation

        self.prepare_dataset(
            sequence_pkl_path
        )

    def get_min_max(self, sequences):
        min, max = 0, 0

        for seq in sequences:
            cur_min = torch.min(seq)
            cur_max = torch.max(seq)

            if cur_min < min:
                min = cur_min

            if cur_max > max:
                max = cur_max

        return min, max


    def get_log_scaled(self, sequences):
        
        for seq in sequences:
            negative_mask = seq < 0
            seq[negative_mask] = -1 * torch.log(-1*seq[negative_mask])
            positive_mask = seq > 0 #  ignore zeros
            seq[positive_mask] = torch.log(seq[positive_mask])

        return sequences


    def prepare_dataset(
        self, 
        pkl_path
    ):
        for is_train in [True, False]:
            sequences = pickle.load(
                open(f"{pkl_path}_{'train' if is_train else 'test'}.pkl", 'rb')
            )
            
            sequences = self.get_log_scaled(sequences)

            if is_train:
                self.train = sequences
            else:
                self.test = sequences

    def make_batch_input_output(self, sequences):
        input_sequences, targets_fc, targets_seq = [], [], []
        sequences.sort(key=len, reverse=True)

        for sequence in sequences:
            input_sequences.append(sequence[:-1])

            targets_fc.append(
                torch.unsqueeze(
                    torch.tensor([sequence[-1, i] for i in self.output_indexes]),
                    dim=1
                )
            )
            targets_seq.append(sequence[1:])


        return (
            pack_sequence(input_sequences),
            targets_fc,
            pad_packed_sequence(pack_sequence(targets_seq))[0]
        )

    def perform_epoch(self, train=True):

        if train:
            self.lstm.train()
        else:
            self.lstm.eval()

        dataset = self.train if train else self.test
        total_fc_loss, total_seq_loss = 0, 0
        prediction, targets_fc = None, None

        for i in range(0, len(dataset), self.batch_size):
            self.optimizer.zero_grad()
            sequence_batch = dataset[i:i+self.batch_size]

            if len(sequence_batch) < self.batch_size:
                sequence_batch = sequence_batch + \
                    random.sample(dataset, self.batch_size - len(sequence_batch))

            input_sequences, targets_fc, targets_seq = self.make_batch_input_output(sequence_batch)

            prediction, sequence_outputs = self.lstm.forward(
                input_sequences.to(self.device),
                self.init_h0, 
                self.init_c0,
                activations=self.activation
            )

            loss_fc = self.loss_function(
                prediction,
                torch.stack(targets_fc, dim=0).squeeze(dim=-1).to(self.device) 
            )

            loss_seq = self.loss_function(
                sequence_outputs[:, :, :self.num_features] + sequence_outputs[:, :, -self.num_features:],
                targets_seq.to(self.device)
            )

            if train:
                loss = loss_fc + loss_seq
                loss.backward()
                self.optimizer.step() 

                # loss_seq.backward()
                # self.optimizer.step() 

            loss_fc, loss_seq = float(loss_fc), float(loss_seq)
            total_fc_loss += loss_fc
            total_seq_loss += loss_seq

        #print("PRED :", prediction, "TRAGET :", targets_fc)
        print(f'\t{"Train" if train else "Validation"} fully connected Loss={total_fc_loss}, sequence loss={total_seq_loss}')
        return total_fc_loss, total_seq_loss

    def save_loss_plot(self, losses, train=True, fig=None):
        if fig is not None:
            fig.clear(True) 
        else:
            fig = plt.figure(figsize=(25,10), clear=True)
        
        ax = fig.add_axes([0.1, 0.1, 0.5, 0.5])
        ax.plot(range(len(losses)), losses, label='train' if train else 'test')
        ax.legend()
        plt.savefig(f"{'train' if train else 'test'}_plot_cnt.png", format='png', dpi=300)
        plt.close('all')

    def looper(self, load_model=False, model_path='lstm_model.pt', opim_path='optimizer.pt'):
        train_losses, test_losses = [], []
        fig = None

        if load_model:
            self.lstm.load_state_dict(torch.load(model_path))
            
        for i in range(3000):
            print(f"Epoch {i}::")
            train_loss = self.perform_epoch()
            test_loss = self.perform_epoch(train=False)

            if i > 2:
                train_losses.append(train_loss)
                test_losses.append(test_loss)

            if i % 20 == 0 and i != 0:
                # in order to modify the size
                self.save_loss_plot(train_losses, True, fig)
                self.save_loss_plot(test_losses, False, fig)

            if i % 100 == 0:
                # Print model's state_dict
                # print("Model's state_dict:")
                # for param_tensor in self.lstm.state_dict():
                #     print(param_tensor, "\t", self.lstm.state_dict()[param_tensor].size())

                torch.save(self.lstm.state_dict(), model_path)
                torch.save(self.optimizer.state_dict(), opim_path)

        pass
