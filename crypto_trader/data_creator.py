# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:16:46 2021

@author: Joe
"""
import datetime
import queue
import time

import cv2
import numpy as np
import pyautogui
import pandas as pd
import d3dshot

from helpers.api_requests import get_qoute
from crypto_trader.data import exponential_moving_average_step

# from binance.client import Client
# from binance.exceptions import BinanceAPIException, BinanceWithdrawException


# client = Client('8puDT4nMbB9V0pOqiHAWP7qOXWzr4dvhGKq7NWtTENccFaKua3uFCAZo7iuYhbhG',
#                 'hFEHSl0gr2cMmD6LYcjDrbLOQdn62CeOBmusOPyDqlianZPjQPM8LzSJ6ViQJl47')

def quote_and_graph_capture(
        cryptos,
        width=892, 
        height=340,
        x_indent=66,
        downsize_fraction=0.5, 
        dual_crypto=False,
        grayscale=False,
        binance=True,
        use_d3dshot=False,
        d3_instance=None):
    
    screen_size = (1920, 1080)
    

    # make a screenshot
    screenshots = []
    
    # y, x
    sizes = (np.array([[150, x_indent],
                      [150, 1025],
                      [640, 1025],
                      [640, x_indent]]) * downsize_fraction).astype(np.int16)
    
    if not dual_crypto:
        sizes = sizes[:2]

    width = int(width*downsize_fraction)
    height = int(height*downsize_fraction)

    #s = time.time()
    # get quote first to have most upto date screenshot
    quote = get_qoute(cryptos, binance=binance, drop_columns=True)
    # print("quote time", time.time() - s)
    
    # print(quote)
    # print(quote.values())
    
    # s = time.time()

    if use_d3dshot:
        screenshot = cv2.cvtColor(
            np.array(d3_instance.screenshot()),
            cv2.COLOR_BGR2GRAY if grayscale else cv2.COLOR_BGR2RGB 
        )
    else:
        screenshot = cv2.cvtColor(
            np.array(pyautogui.screenshot()),
             cv2.COLOR_BGR2GRAY if grayscale else cv2.COLOR_BGR2RGB 
        )
        
    screenshot = cv2.resize(
        screenshot,
        (int(screen_size[0]*downsize_fraction), int(screen_size[1]*downsize_fraction)),
        interpolation = cv2.INTER_AREA
    )
        
    
    for size in sizes:
        sc = screenshot[size[0]:size[0]+height,
                       size[1]:size[1]+width]
        screenshots.append(sc
        )
        
        # cv2.imshow('', sc)
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows() 
    # print(time.time() - s)
        
    return screenshots, quote

# def exponential_moving_average(last_ema, current_value, num_entries, smoothing):
#     return current_value*(smoothing/(1+num_entries)) + last_ema*(1 - (smoothing/(1+num_entries)))


def create_episode_data(num_hours_to_record,
                        width=892,
                        height=340,
                        x_indent=68,
                        downsize_fraction=0.5,
                        grayscale=False,
                        file_name_prefix='data/streams/BTC_episode_data',
                        cryptos='BTCEUR',
                        binance=True,
                        use_d3dshot=False):
    
    avg_process_time = 0.4
    num_captures = int(num_hours_to_record*3600/avg_process_time)
    process_time_ema, smoothing = None, 0.05
    
    dual_crypto = not type(cryptos) == str
    
    image_array = np.zeros([
        num_captures,
        int(height*downsize_fraction),
        int(width*downsize_fraction),
        (4 if grayscale else 12) if dual_crypto else  (2 if grayscale else 6)
    ], dtype='uint8')
    
    d3_instance = None
    if use_d3dshot:
         d3_instance = d3dshot.create(capture_output="numpy")
        
    quote_data = np.zeros(
        [
            num_captures, 
            len(get_qoute(binance=binance, drop_columns=True)),
            2 if dual_crypto else 1
        ] ,
        dtype=np.float32
    )
    start_time = time.time()
    
    file_name_prefix = file_name_prefix+'_'+datetime.datetime.now().strftime("%d_%m_%Y")
    
    click_positions = [[150, 8],
                       [1500, 8]]
                        # [127, 553],
                        # [1500, 553]]
    
    for i in range(num_captures):
        capture_start = time.time()
        
        if i != 0:
            if i % 90 == 0:
                pyautogui.click(click_positions[0])
            elif i % 120 == 0:# 0.12 seconds
                pyautogui.click(click_positions[1])
            # if i % 140 == 0:
            #     pyautogui.click(click_positions[2])
            # elif i % 160 == 0:# 0.12 seconds
            #     pyautogui.click(click_positions[3])
            
        screenshots, quote = quote_and_graph_capture(
            cryptos,
            width,
            height,
            x_indent,
            downsize_fraction,
            dual_crypto,
            grayscale, 
            binance,
            use_d3dshot,
            d3_instance
        )
        
        # print(quote)

        # cv2.imshow('', np.concatenate(screenshots, axis=0))
        # cv2.waitKey(2000)
        # cv2.destroyAllWindows() 
        
        image_array[i, :, :, :] = np.moveaxis(np.array(screenshots), 0, -1)
        
        if dual_crypto:    
            quote_data[i, :, :] = np.swapaxes(
                np.array([
                    list(quote[0].values()),
                    list(quote[1].values())
                ], dtype=np.float32), 0, 1
            )
            
        else:
            quote_data[i, :, :] = np.array(list(quote.values())).reshape([-1, 1])
            
        # print(quote_data[i, :, :])

        process_time = time.time()-capture_start
        
        if process_time_ema is None:
            process_time_ema = process_time
        else:
            process_time_ema = exponential_moving_average_step(
                process_time_ema, process_time, smoothing
            )
            
    
        if i % 10 == 0:
            print(f'Capture {i} of {num_captures}, time elapsed={time.time()-start_time} seconds, average capture time={process_time_ema} seconds')

    process_hours_total = str(round((time.time()-start_time) / 3600, 2)).replace('.', '_')
    t = time.time()
    np.save(file_name_prefix+'_images_{}_hours.npy'.format(process_hours_total), image_array)
    np.save(file_name_prefix+'_quotes_{}_hours.npy'.format(process_hours_total), quote_data)
    print('Save time: ', time.time() - t)


def create_numeric_episode_data(
    num_hours_to_record,
    file_name_prefix='data/streams/BTC_episode_data',
    cryptos='XRPEUR',
    binance=True,
    sleep_time=25
):
    data = pd.DataFrame()
    num_captures = int(num_hours_to_record*3600/sleep_time)
    start_date_time = datetime.datetime.now().strftime("%m_%d_%Y__%H_%M")

    print(f"Performing {num_captures} captures over {num_hours_to_record} hours")

    cur_sleep_time = 1
    thread_queue = queue.Queue()

    # get quote every n seconds
    for i in range(num_captures):
        start_time = time.time()
        
        if cur_sleep_time == 0:
            print("Time exceeded on last iteration.")

        quote = get_qoute(
            cryptos,
            binance=binance,
            drop_columns=True,
            thread_queue=thread_queue
        )

        if len(cryptos) > 0:
            quote_ = {}

            for j in range(len(cryptos)):
                label = quote[j]['ticker'][:3]+"_"
                quote_ = {
                    **quote_,
                    **{
                        label+key: value
                        for key, value in quote[j].items() if key != 'ticker'
                    }
                }

            quote = quote_

        quote['timestamp'] = datetime.datetime.now().strftime("%m:%d:%Y %H:%M:%S")
        data = data.append(quote, ignore_index=True)

        if i % 100 == 0: #500:
            print(f"Capture {i} of {num_captures}, Saving data to disk.")
            data.to_csv(file_name_prefix+start_date_time+".csv") # started og one at 9pm

        cur_sleep_time = max(0, sleep_time - (time.time() - start_time))
        time.sleep(cur_sleep_time)


if __name__ == "__main__":
    
    
    path_prefix = '' #'D:/'
    file_name_prefix = path_prefix + 'data/streams/XRPEUR/numeric/12_second_interval_'   #gui_episode_data'
    load = False
    
    numeric_only = True

    if numeric_only:
        create_numeric_episode_data(
            num_hours_to_record=800,
            file_name_prefix=file_name_prefix,
            cryptos=['XRPEUR', 'BTCEUR'],
            binance=True,
            sleep_time=12
        )
    else:
    
        if load:    
            hours = '10_27'
            date = "05_04_2021"
                
            images_array = np.load(
                file_name_prefix+'_{}_images_{}_hours.npy'.format(date, hours)
            )
            
            # quotes_array = np.load(
            #     file_name_prefix+'_{}_quotes_{}_hours.npy'.format(date, hours)
            # )        
            

            for i in range(0, images_array.shape[0], 1):
                cv2.imshow('', np.concatenate([images_array[i, :, :, 0],
                                                images_array[i, :, :, 1]], axis=0))
                cv2.waitKey(1500)
                cv2.destroyAllWindows() 
                
            # np.save(
            #     file_name_prefix+'_{}_images_{}_hours_mod.npy'.format(date, hours),
            #     images_array[:97286]
            # )
            # np.save(
            #     file_name_prefix+'_{}_quotes_{}_hours_mod.npy'.format(date, hours),
            #     quotes_array[:97286]
            # )
        
        
        create_episode_data(
            num_hours_to_record=12,
            x_indent=60,
            downsize_fraction=0.4,
            grayscale=True,
            file_name_prefix=file_name_prefix,
            cryptos='XRPEUR',
            use_d3dshot=False,
            binance=True
        )    
    
    
    
    
    
    
