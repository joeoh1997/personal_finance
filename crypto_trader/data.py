# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:09:13 2021

@author: Joe
"""
import os
from copy import deepcopy

import numpy as np
import pandas as pd


'''
Quote:
    {'priceChange': '0.10264000', 'priceChangePercent': '13.584',
     'weightedAvgPrice': '0.80952910', 'prevClosePrice': '0.75556000',
     'lastPrice': '0.85821000', 'lastQty': '6.90000000',
     'bidPrice': '0.85686000', 'bidQty': '22430.50000000',
     'askPrice': '0.85804000', 'askQty': '860.00000000',
     'openPrice': '0.75557000', 'highPrice': '0.86235000',
     'lowPrice': '0.75021000', 'volume': '33943299.10000000',
     'quoteVolume': '27478088.38265500', 'count': 72506}
'''

def normalize_quote_data(
        quote,
        percentage_denom=100,
        volume_denom=100000000,
        bid_ask_denom=10000
    ):
    
    headers = [
         'PriceChange', 'PriceChangePercent',
         'weightedAvgPrice', 'prevClosePrice',
         'lastPrice', 'lastQty',
         'bidPrice', 'bidQty',
         'askPrice', 'askQty',
         'openPrice', 'highPrice',
         'lowPrice', 'volume',
         'quoteVolume', 'count'
    ]
    drop_columns = ['PriceChange'] + headers[-2:]
    
    # convert quote data to dataframe
    df = pd.DataFrame(data=quote.squeeze(axis=-1), columns=headers).drop(drop_columns, axis=1)

    # volume norm
    df['volume'] = df['volume']/volume_denom
    
    # get price headers
    price_headers = [header for header in headers if 'Price' in header]
    [price_headers.remove(var) for var in [
        'PriceChangePercent', 'PriceChange', 'weightedAvgPrice'
    ]]
    
    # normalize price to distance from weighted avg
    df[price_headers] = df[price_headers].sub(df['weightedAvgPrice'], axis=0)   
    weightedAvgPrice = deepcopy(df['weightedAvgPrice']).to_numpy()
    df = df.drop('weightedAvgPrice', axis=1)    
    
    # normalize vars using denom
    df[['bidQty', 'askQty', 'lastQty']] = df[['bidQty', 'askQty', 'lastQty']]/bid_ask_denom 
    df['PriceChangePercent'] = df['PriceChangePercent']/percentage_denom

    return (
        df.to_numpy(np.float16),
        quote[:, 4].astype(np.float32).flatten(),
        weightedAvgPrice.astype(np.float32).flatten()
    )


def get_stream_data_sizes(file_path='data/streams/XRPEUR/', half=False):
    """
         Function to get number of quote variable & 
         Size of image data
    """
    
    quote_filenames = [path for path in os.listdir(file_path) if 'quote' in path]
    image_filenames = [path for path in os.listdir(file_path) if 'image' in path]
    
    quote_array = np.load(file_path+quote_filenames[0])[0]
    image_array = np.load(file_path+image_filenames[0])[0]
    
    quote_shape = quote_array.flatten().shape[0]
    image_array_shape = image_array.shape
    
    quote_array, image_array = None, None
    
    stream_dates = [filename.split('_')[3:6] for filename in quote_filenames]
    stream_time = [filename.split('_')[7:9] for filename in quote_filenames]
    
    if half:
        image_array_shape = [image_array_shape[0],
                             int(image_array_shape[1]/2)-40,
                             image_array_shape[2]]
        
    return quote_shape, image_array_shape, stream_dates, stream_time


def get_stream_data(stream_date,
                    stream_time,
                    file_name_prefix='gui_episode_data',
                    file_path='data/streams/XRPEUR/',
                    half=False):
    """
         Function to get quote & image stream data
         from specified stream.
    """
    quote_array = np.load(
        '{}{}_{}_{}_{}_quotes_{}_{}_hours.npy'.format(
            file_path, file_name_prefix, *stream_date, *stream_time
        )
    )
    
    image_array = np.load(
        '{}{}_{}_{}_{}_images_{}_{}_hours.npy'.format(
            file_path, file_name_prefix, *stream_date, *stream_time
        )
    )
    
    if half:
        image_array = image_array[:, :, int(image_array.shape[2]/2):-40, :]

    return quote_array, image_array


def exponential_moving_average_step(last_ema, current_value, smoothing):
    """
        Function gets 1 ema step value
    """
    return current_value*smoothing + last_ema*(1 - smoothing)


def prep_sim_data(sim_index, stream_dates, stream_times, data_path):
    """
        This functions prepares saved sim data for learning

    Parameters
    ----------
        sim_index : int index of stream file.
        stream_dates : list of sim stream dates.
        stream_times : list of sim stream durations.
        data_path : string path to data directory

    """
    stream_date = stream_dates[sim_index]
    stream_time = stream_times[sim_index]
    
    sim_numeric_data, sim_image_data = get_stream_data(
        stream_date,
        stream_time,
        file_path=data_path,
        half=True
    )
    
    sim_numeric_data, sim_prices, weightedAvgPrice = normalize_quote_data(
        sim_numeric_data
    )
          
    buy_rewards = weightedAvgPrice - sim_prices  # positive if buy price less than average price
    
    return sim_numeric_data, sim_image_data, buy_rewards, sim_prices, weightedAvgPrice
    
    
    