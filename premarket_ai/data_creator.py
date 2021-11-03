# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:16:46 2021

@author: Joe
"""
import webbrowser
from datetime import datetime, date
import time
import os
import sys
import ast

#import cv2
import numpy as np
#import pyautogui
import pandas as pd
from PIL import Image

from helpers.api_requests import get_premarket_movers, get_minute_price_data

from premarket_ai.api_calls import pipeline



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


def get_premarket_charts(tickers, save_dir, todays_date_str):
    # get chart screenshots
    for ticker in tickers+['SPY', 'NQ00']:
        premarket_chart = open_website_capture_screnshot(
                'https://www.marketwatch.com/investing/stock/'+ticker,
                region=(340, 200, 1005, 800),
                delay_before_capture=5,
                plot=False
        )
        Image.fromarray(premarket_chart).save(
            "{}{}_{}.png".format(save_dir, ticker, todays_date_str)
        )
        

def get_finviz_charts(tickers, save_dir, todays_date_str):
    # get chart screenshots
    for ticker in tickers+['NQ00']:
        daily_chart = open_website_capture_screnshot(
            'https://finviz.com/quote.ashx?t='+ticker,
            region=(230, 383, 1440, 400),
            delay_before_capture=2,
            plot=False
        )    
        
        Image.fromarray(daily_chart).save(
            "{}{}_{}.png".format(save_dir, ticker, todays_date_str)
        )
    
    
def get_premarket_movers_data_and_images(
        num_movers=12,
        path='data/premarket_movers/movers'
    ):
    last_day = max([int(pth) for pth in os.listdir(path)])
    path = f"{path}/{last_day+1}/"
    os.makedirs(path)

    print(f"Last day :{last_day}, path :{path}\n")
        
    movers = get_premarket_movers(num_movers)
    pd.DataFrame(movers).to_CSV(f"{path}_movers.csv")
    
    tickers = list(movers.keys())

    pipeline(tickers, path)
        
    return tickers


def get_premarket_movers_data_and_images_timed(
        num_movers=12,
        trigger_hour=13,
        trigger_min=0,
        trigger_sec=0
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
    
    tickers = get_premarket_movers_data_and_images(num_movers)
    
    print(f"Got movers, Start: {read_time}",
          f", End: {datetime.now().strftime('%H:%M:%S')}")
    print('Tickers: "{}"'.format(tickers))
    
    
def get_premarket_mover_market_open_data(
        tickers,
        trigger_hour=15,
        trigger_min=40,
        trigger_sec=0,
        path='data/premarket_movers/movers'):

    last_day = max([int(pth) for pth in os.listdir(path)])
    path = f"{path}/{last_day}/"

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

    market_tickers = [
        "^FVX","^TNX", "^TYX",
        "^VIX", "^OVX", "^GVZ",
        "BTCUSD", "EURUSD", 
        "GCUSD", "CLUSD",
        "^GSPC", "^DJI", "^IXIC"
    ]
        
    while(datetime.now() <= read_time):
        pass
    
    start_readtime = datetime(todays_date.year,
                              todays_date.month,
                              todays_date.day,
                              9, 29, 59)
    end_readtime = datetime(todays_date.year,
                            todays_date.month,
                            todays_date.day,
                            16, 1, 1)
    
    for ticker in tickers + market_tickers:
        pd.DataFrame(get_minute_price_data(
            ticker, start_readtime, end_readtime, five_min=True
        )).to_csv(
            "{}{}_market_open_price.csv".format(path, ticker),
            index=False
        )
    

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        get_premarket_mover_market_open_data(ast.literal_eval(sys.argv[1]))
    else:
        get_premarket_movers_data_and_images_timed()

    

    

        
    
    

    
    
    
    
