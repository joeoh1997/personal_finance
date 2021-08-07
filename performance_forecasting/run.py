# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:22:43 2021

@author: joeoh
"""
import torch 
import torch.nn.functional as functional

from performance_forecasting.data_creator import download_statement_data_for_exchanges, save_sequences_to_disk
from performance_forecasting.trainer import Trainer

selected_variables = [ 
  'revenue', 'costOfRevenue', 'grossProfit', 'researchAndDevelopmentExpenses',
  'sellingAndMarketingExpenses',  'operatingExpenses', 'operatingIncome',
  'investmentsInPropertyPlantAndEquipment', 'debtRepayment', 'commonStockIssued',
  'commonStockRepurchased', 'capitalExpenditure', 'freeCashFlow'
 ]

variables_to_forecast = ['revenue', 'freeCashFlow']

period = 'year'
min_statements = 4
max_years=30

download_statements = False 
create_sequences = False
train = True 

path_prefix = '' #'D:/'
data_path = f"{path_prefix}data/statements/{period}/"
pkl_path='aggregation/sequences'

exchanges = ['EURONEXT' ,'NYSE', 'NASDAQ', 'LSE']


if download_statements:
    download_statement_data_for_exchanges(
        exchanges, period, min_statements, data_path
    )

if create_sequences:
    save_sequences_to_disk(
        data_path=data_path,
        selected_variables=selected_variables,
        min_statements=min_statements,
        max_years=max_years,
        pkl_path=pkl_path
    )


layer_sizes = {
    'lstm_hidden_size': int(len(selected_variables)*2*1.5),
    'fc_hidden_size': 2,
    'num_stacked_lstm_cells': 6
}

if train:
    trainer = Trainer(
        data_path+pkl_path,
        variables_to_forecast,
        selected_variables,
        layer_sizes,
        weight_decays={
            'lstm_dropout': 0,  #0.333
            'fc_l2': 0.01,  # 0.001
            'global_l2': 0.0001  # 0.0001
        },
        lr=0.001,
        batch_size=45, # update params
        optim='asgd',
        loss_function='l2',
        activation=[torch.tanh, None], # functional.softsign
        use_bn=False,
        use_rnn=False # network config
    )

    trainer.looper(load_model=False)