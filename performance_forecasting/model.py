# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:22:43 2021

@author: joeoh
"""
import torch
import torch.nn as nn
#import torch.nn.utils.rnn.pad_packed_sequence as pad_packed_sequence

#from torch import nn.utils.rnn.pad_packed_sequence as pad_packed_sequence

class revenueForecastLSTM(nn.Module):
    def __init__(self, num_features, num_outputs, layer_sizes, use_bn=True, use_rnn=False):
        super(revenueForecastLSTM, self).__init__()

        m = nn.RNN if use_rnn else nn.LSTM 
        self.num_features = num_features

        self.lstm = m(
            input_size=num_features,
            hidden_size=layer_sizes['lstm_hidden_size'],
            num_layers=layer_sizes['num_stacked_lstm_cells'],
            batch_first=True,  # if True input size = (batch, seq, feature) otherwise seq first
            dropout=0,
            bidirectional=False
        )
        self.bn_0 = nn.BatchNorm1d(layer_sizes['lstm_hidden_size']*(1 if use_rnn else 2))
        self.prelu_fc_0 = nn.PReLU()

        self.fully_connected_0 = nn.Linear(
            layer_sizes['lstm_hidden_size']*(1 if use_rnn else 2),
            layer_sizes['fc_hidden_size']
        )
        self.prelu_fc_1 = nn.PReLU()
        self.bn_1 = nn.BatchNorm1d(layer_sizes['fc_hidden_size'])

        self.fully_connected_1 = nn.Linear(
            layer_sizes['fc_hidden_size'], num_outputs
        )

        #self.prelu_fc_1 = nn.PReLU()
        #self.bn_1 = nn.BatchNorm1d(layer_sizes['fc_hidden_size'])

        self.initialize_lstm_weights(layer_sizes)

        self.use_rnn = use_rnn
        self.use_bn = use_bn

        # build actual NN
        #self.__build_model()

    def initialize_lstm_weights(self, layer_sizes):
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.ones_(
            self.lstm.bias_ih_l0[layer_sizes['lstm_hidden_size']:layer_sizes['lstm_hidden_size']*2]
        )
        
        nn.init.zeros_(self.lstm.bias_hh_l0)
        nn.init.ones_(
            self.lstm.bias_hh_l0[layer_sizes['lstm_hidden_size']:layer_sizes['lstm_hidden_size']*2]
        )

    def forward(self, packed_sequences, init_hidden_state, init_output_state, activation=None):
        """
            Just training on final output, only final output usedf for prediction
        """

        if self.use_rnn:
            sequence_outputs, hidden_state = self.lstm(packed_sequences, init_hidden_state)
            x = hidden_state

        else:
            sequence_outputs, (hidden_state, output_state) = self.lstm(
                packed_sequences, (init_hidden_state, init_output_state)
            )
            x = torch.cat([hidden_state, output_state], axis=-1)

        activation = activation if activation else self.prelu_fc_0
        x = activation(torch.swapdims(x, 0, 1).sum(dim=1))

        if self.use_bn:
            x = self.bn_0(x)

        activation = activation if activation else self.prelu_fc_1
        x = activation(self.fully_connected_0(x))

        if self.use_bn:
            x = self.bn_1(x)

        prediction = self.fully_connected_1(x)  #self.batch_norm_0(

        return prediction, torch.nn.utils.rnn.pad_packed_sequence(sequence_outputs)[0]


        
 