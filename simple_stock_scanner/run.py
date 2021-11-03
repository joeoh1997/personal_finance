# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 18:05:16 2021

@author: Joe
"""
#ToDo: Add total debts linear model (found in balance sheet)

from performance_forecasting.data_creator import download_statement_data_for_exchanges
from simple_stock_scanner.scanner import create_linear_models_from_statements, save_financial_ratios, save_merged_ratio_and_model_data

period = 'yearly'
min_statements = 5
max_years=30

download_statements = False
save_linear_models = False
save_ratios = False

exchanges = ['EURONEXT' ,'NYSE', 'NASDAQ', 'LSE']

selected_linear_model_variables = [ 
  'revenue', 'operatingIncome', 'grossProfit', 'netIncome', 'freeCashFlow',
  'investmentsInPropertyPlantAndEquipment', 'debtRepayment', 'commonStockIssued',
  'commonStockRepurchased', 'capitalExpenditure'
 ]

selected_ratios = [
  "debtRatioTTM", "debtEquityRatioTTM", "longTermDebtToCapitalizationTTM",
  "totalDebtToCapitalizationTTM", "interestCoverageTTM",
  "cashFlowToDebtRatioTTM", "priceBookValueRatioTTM",
  "priceToBookRatioTTM", "priceToSalesRatioTTM",
  "priceEarningsRatioTTM", "priceToFreeCashFlowsRatioTTM",
  "priceToOperatingCashFlowsRatioTTM", "priceCashFlowRatioTTM",
  "priceEarningsToGrowthRatioTTM", "priceSalesRatioTTM",
]

path_prefix = ''
data_path = f"{path_prefix}data/statements/{period}/"
linear_model_folder_name = 'linear_model_data'

if download_statements:
    download_statement_data_for_exchanges(exchanges, period, min_statements, data_path)
    
if save_linear_models:
    create_linear_models_from_statements(
        data_path,
        selected_linear_model_variables,
        num_years_to_model=6
    )

if save_ratios:
    save_financial_ratios(
        data_path
    )

save_merged_ratio_and_model_data(
    data_path,
    selected_ratios
)
