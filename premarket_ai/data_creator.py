# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:16:46 2021

@author: Joe
"""
import webbrowser
from datetime import datetime, date, timedelta
import time
import requests
import os
import sys
import ast

import cv2
import numpy as np
import pyautogui
import pandas as pd
from PIL import Image
from urllib.request import Request, urlopen

from helpers.api_requests import get_premarket_movers, get_qoute, financial_model_prep_growth, get_minute_price_data
from helpers.linear_forecasts import get_linear_model_data_from_statement



def open_website_capture_screnshot(
        url,
        region=(0, 0, 1920, 1080),
        browser_exe_address="C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        grayscale=False,
        delay_before_capture=3,
        plot=False
    ):

    webbrowser.register('chrome',
     	None,
     	webbrowser.BackgroundBrowser(browser_exe_address))
    webbrowser.get('chrome').open(url)
            
    time.sleep(delay_before_capture)    
    screenshot = np.array(pyautogui.screenshot(region=region))
    
    if plot:
        cv2.imshow('', screenshot)
        cv2.waitKey(2000)
        cv2.destroyAllWindows() 
        
    return screenshot
    
    
def get_premarket_movers_data_and_images(
        num_movers=12,
        data_filename='data/premarket_movers/premarket_mover_info.csv',
        premarket_chart_folder='data/premarket_movers/premarket_charts/',
        daily_chart_folder='data/premarket_movers/daily_charts/'
    ):
    
    quote_yesterday_headers = ['changesPercentage',
                               'change',
                               'dayLow',
                               'dayHigh',
                               'volume',
                               'open'] 
    quote_change_headers = {'price': 'yesterday_close',
                            'dayHigh_diff': 'yesterday_high_close_diff',
                            'dayLow_diff': 'yesterday_low_close_diff',
                            'yearHigh_diff': 'year_high_yesterday_close_diff',
                            'yearLow_diff': 'year_close_yesterday_close_diff',
                            'previousClose': 'close_day_before_yesterday'}
    
    quote_drop_columns = ['earningsAnnouncement', 'timestamp']
    
    
    quote_change_headers.update({
         name: 'yesterday_'+name for name in quote_yesterday_headers
    })
        
    movers = get_premarket_movers(num_movers)
    
    data_dict = {} 
    todays_date = date.today()
    todays_date_str = todays_date.strftime("%d_%m_%Y")
    
    # get linear model statement growth
    for ticker in list(movers.keys()):
        try:
            weights = get_linear_model_data_from_statement(
                ticker,
                'quarter',
                ['revenue'],
                start_date=datetime.strptime('2014-12-31', '%Y-%m-%d').date(),
                plot=False
            )
            if weights:
                data_dict[ticker] = weights
            
        except:
            print('Error adding linear models for {}...'.format(ticker))
            
    # get tickers which we have statement data for
    tickers = list(data_dict.keys())
    
    # get chart screenshots
    for ticker in tickers+['SPY', 'NQ00']:
        premarket_chart = open_website_capture_screnshot(
                'https://www.marketwatch.com/investing/stock/'+ticker,
                region=(340, 200, 1005, 800),#555
                delay_before_capture=5,
                plot=False
        )
        Image.fromarray(premarket_chart).save(
            "{}{}_{}.png".format(premarket_chart_folder, ticker, todays_date_str)
        )
        
        if ticker != 'NQ00':
            daily_chart = open_website_capture_screnshot(
                'https://finviz.com/quote.ashx?t='+ticker,
                region=(230, 383, 1440, 400),
                delay_before_capture=2,
                plot=False
            )    
            
            Image.fromarray(daily_chart).save(
                "{}{}_{}.png".format(daily_chart_folder, ticker, todays_date_str)
            )
        
        
    # quote doesnt include premarket price
    for qoute in get_qoute(tickers,
                           add_differences=True,
                           drop_columns=quote_drop_columns):
        
        data_dict[qoute['symbol']] = {
            **data_dict[qoute['symbol']],
            **qoute,
            **financial_model_prep_growth(qoute['symbol'])[0]
        }
        
        
    data_df = pd.DataFrame(data_dict).transpose()
    data_df['growth_calculation_date'] = data_df['date']
    data_df['date'] = [todays_date]*len(data_df)
    
    data_df = pd.concat([
        data_df.rename(quote_change_headers, axis=1),
        pd.DataFrame(movers).transpose().drop('name', 1)
    ], axis=1)
    
    if os.path.isfile(data_filename):
        previous_data = pd.read_csv(data_filename, index_col=False).append(
              data_df
        ).to_csv(data_filename, index=False)
        
    else:
        data_df.to_csv(data_filename, index=False)
        
    return tickers


def get_premarket_movers_data_and_images_timed(
        num_movers=12,
        trigger_hour=14,
        trigger_min=26,#16,#28,
        trigger_sec=30,#0,
        market_open_folder='data/premarket_movers/market_open_data/'
    ):
    
    todays_date = date.today()
    read_time = datetime(
        todays_date.year,
        todays_date.month,
        todays_date.day,
        trigger_hour,
        trigger_min,
        trigger_sec
    )
        
    print('Started, waiting until', read_time.strftime("%H:%M:%S"))
    
    while(datetime.now() <= read_time):
        pass
    
    tickers = get_premarket_movers_data_and_images(num_movers) + ['SPY']
    
    print(f"Got movers, Start: {read_time}",
          f", End: {datetime.now().strftime('%H:%M:%S')}")
    print('Tickers: "{}"'.format(tickers))
    
    
def get_premarket_mover_market_open_data(
        tickers,
        trigger_hour=15,
        trigger_min=40,
        trigger_sec=0,
        market_open_folder='data/premarket_movers/market_open_data/'):

    todays_date = date.today()
    read_time = datetime(
        todays_date.year,
        todays_date.month,
        todays_date.day,
        trigger_hour,
        trigger_min,
        trigger_sec
    )
    print('Started, waiting until', read_time.strftime("%H:%M:%S"))
        
    while(datetime.now() <= read_time):
        pass
    
    start_readtime = datetime(todays_date.year,
                              todays_date.month,
                              todays_date.day,
                              9, 29, 59)
    end_readtime = datetime(todays_date.year,
                            todays_date.month,
                            todays_date.day,
                            10, 30, 1)
    
    for ticker in tickers:
        pd.DataFrame(
            get_minute_price_data(ticker, start_readtime, end_readtime)
        ).to_csv(
                "{}{}_{}.csv".format(market_open_folder,
                                     ticker,
                                     todays_date.strftime("%d_%m_%Y")),
                index=False
            )
    

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        get_premarket_mover_market_open_data(ast.literal_eval(sys.argv[1]))
    else:
        get_premarket_movers_data_and_images_timed()

    

    

        
    
    

    
    
    
    
