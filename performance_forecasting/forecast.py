# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:20:17 2021

@author: joeoh
"""

from helpers.api_requests import linear_forecasts


from datetime import datetime

l = linear_forecasts.get_linear_model_data_from_statement(
    ticker='FB',
    period='quarter',
    varnames='all',
    start_date='2008-12-31',
    end_date=None,
    plot=True,
    model_data=None
)

print(l)
# '%Y-%m-%d'
# f = linear_forecasts.get_price_data_between_date_range(
#         ticker='FB',
#         start_date,
#         forecast_date,
#         end_date,
#         price_data=None
# )