# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:06:38 2021

@author: Joe
"""
import os
import pickle

import pandas as pd

from performance_forecasting.data_creator import create_linear_model_data
from helpers.api_requests import get_financial_ratios


def create_linear_models_from_statements(
        data_path,
        selected_variables, 
        num_years_to_model=5
    ):
    """
    Creates linear models for each selected var from each statements csv saved in directory
    """
    linear_model_data = pd.DataFrame()
    paths = os.listdir(data_path)

    for i, statement_path in enumerate(paths):
        try:
            statement_df = pd.read_csv(
                data_path+statement_path, index_col=0
            ).loc[:num_years_to_model, :]

            ticker = statement_df['symbol'].values[0]
            statement_df = statement_df[selected_variables]

            linear_model_data = linear_model_data.append(
                create_linear_model_data(statement_df, ticker=ticker, plot=False),
                ignore_index=True
            )

            if i % 500 == 0:
                print(f"processed {i} out of {len(paths)} tickers")
                linear_model_data.to_csv(data_path+'linear_model_data.csv')

        except Exception as e:
            print(f'Error while reading path:{statement_path}. Error msg: {e}')

    linear_model_data.to_csv(data_path+'linear_model_data.csv')


def save_financial_ratios(
    data_path
):
    ratios = pd.DataFrame()
    paths = os.listdir(data_path)
    ticker = ""

    linear_model_data = pd.read_csv(data_path+'linear_model_data.csv')

    for i, ticker in enumerate(linear_model_data['ticker'].values):
        try:
            ticker = ticker[:-1] if ticker[-1] == '.' else ticker
            ticker_ratios = get_financial_ratios(ticker)
            
            ratios = ratios.append(
                {**{"ticker": ticker}, **ticker_ratios},
                ignore_index=True
            )

            if i % 500 == 0:
                print(f"processed {i} out of {len(paths)} tickers")
                ratios.to_csv(data_path+'ratios.csv')

        except Exception as e:
            print(f'Error processing {ticker}. Error msg: {e}')

    ratios.to_csv(data_path+'ratios.csv')


def save_merged_ratio_and_model_data(
    data_path,
    selected_ratios
):
    linear_model_data = pd.read_csv(data_path+'linear_model_data.csv', index_col=0)
    
    ratios = pd.read_csv(
        data_path+'ratios.csv',
        index_col=0
    ).loc[:, ["ticker"]+selected_ratios]

    linear_model_data.merge(
        ratios, 
        how='left', 
        on="ticker"
    ).to_csv(data_path+'ratio_and_model_data.csv')


def price_to_x_ratio(share_price, x, num_shares):
    """
    Creates ratios eg. price to earnings (net income) ratio
    """
    ratio = None
    try:
        x_per_share = x/num_shares
        ratio = share_price/x_per_share

    except Exception as e:
        print("Error creating ratio. Error message:"+str(e))
    
    return ratio 
