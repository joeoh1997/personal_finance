# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:22:43 2021

@author: joeoh
"""
import torch
import random
import os
import pickle
from matplotlib.pyplot import axis 

import pandas as pd
import numpy as np

from helpers.linear_forecasts import get_statement_data, add_linear_forecast_params
from helpers.api_requests import get_all_tickers_by_exchange


def download_statement_data_for_exchanges(exchanges, period, min_statements, data_path):
    """
        For given exchanges, downloads all tickers then all 
        statements (merge of cash flow & income) for those tickers
    """
    tickers = []
    [tickers.extend(get_all_tickers_by_exchange(exchange=exchange)) for exchange in exchanges]
    tickers = list(set(tickers))

    print('Num tickers = ', len(tickers))

    for ticker in tickers:
        print(f"processing {ticker}, {tickers.index(ticker)} out of {len(tickers)}")
        try:
            statements = get_statement_data(
                    ticker,
                    period,
                    as_dataframe=True,
                    merge_statements=True,
                    min_statements=min_statements,
            )

            if period == "yearly" and \
                statements.astype({'date': str}).date.str.contains(
                    statements.loc[2, 'date'].split('-')[0]
                ).sum() > 1 : 

                print("Print statements not in correct format (yearly).")

            else:
                statements.to_csv(f"{data_path}{ticker.replace('.', '_')}.csv")

        except Exception as e:
            print(e)


def remove_nans_from_statements(statements):
    """
        Removes nans from statments, such that if one statement has nans,
         all previous statements are removed from the data.
    """
    if np.sum(np.isnan(statements)) > 0:
        statements_seqential_no_nans = []

        for statement in statements:
            if np.sum(np.isnan(statement)):
                statements_seqential_no_nans = []
            else:
                statements_seqential_no_nans.append(list(statement))

        statements = np.array(statements_seqential_no_nans)

    return statements


def get_all_sequences_from_statements(statements, min_statements):
    """
        Gets all sequences of statements. 
        Example:
            	min_statements = 2

            	input = [[2019, 10000, 0, -23000],
                         [2018, 7000, 0, -43000],
                         [2017, 6000, 0, -53000],
                         [2016, 5000, 0, -63000]] 

                sequences = [[[2017, 6000, 0, -53000],
                              [2016, 5000, 0, -63000]],

                             [[2018, 7000, 0, -43000],
                              [2017, 6000, 0, -53000],
                              [2016, 5000, 0, -63000]], 

                             [[2019, 10000, 0, -23000],
                              [2018, 7000, 0, -43000],
                              [2017, 6000, 0, -53000],
                              [2016, 5000, 0, -63000]]]
    """
    sequences = []
    end_index = min_statements

    while end_index <= statements.shape[0]:
        sequences.append(torch.FloatTensor(statements[:end_index].copy()))
        end_index += 1

    if statements.shape[0] >= int(min_statements*2):
        sequences.append(
            torch.FloatTensor(statements[-int(statements.shape[0]/2):].copy())
        )

    return sequences


def save_sequences_to_disk(
    data_path,
    selected_variables,
    min_statements,
    max_years,
    pkl_path='aggregation/sequences',
    train_split=0.8
):
    """
        Iterates through statements & creates multiple sequences from each tickers statements
    """
    train, test = [], []
    paths = os.listdir(data_path)
    num_tickers = len(paths)
    max_years_observed = 0
    train_indexes = random.sample(range(num_tickers), int(train_split*num_tickers))

    for i, statement_path in enumerate(paths):
        try:
            statement_df = pd.read_csv(data_path+statement_path, index_col=0)
            statements = statement_df[selected_variables].values

            if statements.shape[0] > max_years_observed:
                max_years_observed = statements.shape[0]

            statements = remove_nans_from_statements(
                np.flip(statements[:max_years, :], axis=0)
            )

            sequences = get_all_sequences_from_statements(
                statements, min_statements
            )

            if i in train_indexes:
                train.extend(sequences)
            else:
                test.extend(sequences)

            if i % 100 == 0:
                print(f"processed {i} out of {num_tickers} tickers")

        except Exception as e:
            print(f'Error while reading path:{statement_path}. Error msg: {e}')

    print(f"Max years observed = {max_years_observed}")
    # [
    #     pickle.dump(dataset, open(f"{data_path}{pkl_path}_{set_name}.pkl", 'wb')) 
    #         for set_name, dataset in {'train': train, 'test': test}.items()
    # ]

    

def get_key(ticker, target_index):
    return f"{ticker}__{target_index}"


def create_next_year_target(statement_df, next_year_index, ticker):
    """
        Gets target (label/output) for model. 
        Composed of:
            * Full statement for 'next' year
            * Change in statement between 'this' & 'next' year 

    """
    label_statement_df = statement_df.iloc[next_year_index]
    year_change_statement = (
        statement_df.iloc[next_year_index] - statement_df.iloc[next_year_index-1]
    ).add_prefix('change_')

    label_statement_df['ticker'] = get_key(ticker, next_year_index)
    return pd.concat([label_statement_df, year_change_statement])


def create_this_year_model_data(statement_df, next_year_index, ticker):
    model_data = {'ticker': get_key(ticker, next_year_index)}
    this_year_data = statement_df[:next_year_index]

    for column in statement_df.columns:
        model_data = add_linear_forecast_params(
            this_year_data[column].values,
            model_data,
            column,
            add_forecast=True,
            plot=True
        )

    return model_data


def make_linear_model_datset(data_path, min_statements, selected_variables, save_folder='aggregation'):
    dataset = pd.DataFrame()
    targets = pd.DataFrame()

    for statement_path in os.listdir(data_path):
        
        ticker = statement_path.split('.')[0]
        statement_df = pd.read_csv(data_path+statement_path, index_col=0)
        statement_df = statement_df[selected_variables]
        end_index = min_statements - 1

        while end_index < len(statement_df):

            dataset = dataset.append(
                create_this_year_model_data(statement_df, end_index, ticker),
                ignore_index=True
            )  # get statements up to simulated 'this' year

            targets = targets.append(
                create_next_year_target(statement_df, end_index, ticker),
                ignore_index=True
            )  # get labels for simulated 'next' year (forecasting)

            end_index += 1

    targets.to_csv(data_path+save_folder+'/targets.csv')
    dataset.to_csv(data_path+save_folder+'/dataset.csv')


"""

Possible Variables:: 
[ 'revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio', 'researchAndDevelopmentExpenses',
  'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 'otherExpenses', 'operatingExpenses',
  'costAndExpenses', 'interestExpense', 'ebitda', 'ebitdaratio', 'operatingIncome',
  'operatingIncomeRatio', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeBeforeTaxRatio',
  'incomeTaxExpense', 'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut',
  'weightedAverageShsOutDil', 'reportedCurrency',
  'acceptedDate', 'period', 'netIncome', 'depreciationAndAmortization', 'deferredIncomeTax', 'stockBasedCompensation',
  'changeInWorkingCapital', 'accountsReceivables', 'inventory', 'accountsPayables', 'otherWorkingCapital',
  'otherNonCashItems', 'netCashProvidedByOperatingActivities', 'investmentsInPropertyPlantAndEquipment',
  'acquisitionsNet', 'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites',
  'netCashUsedForInvestingActivites', 'debtRepayment', 'commonStockIssued', 'commonStockRepurchased',
  'dividendsPaid', 'otherFinancingActivites', 'netCashUsedProvidedByFinancingActivities',
  'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod',
  'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow']

"""