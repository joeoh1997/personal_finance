# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:16:46 2021

@author: Joe
"""
import time

import numpy as np
import pandas as pd 

from helpers.api_requests import get_qoute


import torch


def reward_function(balance,
                    next_exchange_price,
                    ticker,
                    base_balance_at_last_buy,
                    initial_base_balance=1000):
    """
    function which determines reward of owning this wallet

    """
    exchange_asset, base_asset = ticker[:3], ticker[3:]
    
    base_balance_at_sell = get_base_balance(
        balance, next_exchange_price, ticker
    )
    
    
    # return ((
    #     balance[base_asset] + balance[exchange_asset]*next_exchange_price
    # ) - initial_base_balance)/ initial_base_balance
    
    print('\t\t Balance at last buy={}, balance now={}, reward={}'.format(
        base_balance_at_last_buy,
        base_balance_at_sell,
        base_balance_at_sell - base_balance_at_last_buy
    ))

    return base_balance_at_sell - base_balance_at_last_buy # *100


def simulated_market_buy(
        balance,
        ticker,
        sell_fraction,
        next_exchange_price,
        fee=0.001,
        allowance_fraction=0.99
    ):
    
    exchange_asset, base_asset = ticker[:3], ticker[3:]
    
    # quantity of sell asset to sell
    spent_base = allowance_fraction * (sell_fraction * balance[base_asset])
    
    # Sell the asset ...
    
    # Amount of buy asset recieved
    recieved_exchange = (spent_base - spent_base*fee) / next_exchange_price
    
    # Update balance
    balance[exchange_asset] = balance[exchange_asset] + recieved_exchange
    balance[base_asset] = balance[base_asset] - spent_base
    
    return balance


def simulated_market_sell(
        balance,
        ticker,
        sell_fraction,
        next_exchange_price,
        fee=0.001,
        allowance_fraction=0.99
    ):
    
    exchange_asset, base_asset = ticker[:3], ticker[3:]
    
    # quantity of sell asset to sell
    spent_exchange = allowance_fraction * (sell_fraction * balance[exchange_asset])
    
    # Sell the asset ...
    
    # Amount of buy asset recieved
    recieved_base = spent_exchange * next_exchange_price
    recieved_base = recieved_base - (fee*recieved_base)
    
    # Update balance
    balance[base_asset] = balance[base_asset] + recieved_base
    balance[exchange_asset] = balance[exchange_asset] - spent_exchange
    
    return balance


def get_negative_frequency_reward(step_dict, weighting=1000):
    if step_dict['current_step'] == 0:
        reward = 0
    else:
        reward = (
            -1*weighting/(step_dict['current_step'] - step_dict['last_buy_atempt_step']) 
            -1*weighting/(step_dict['current_step'] - step_dict['last_sell_attempt_step'])
        )

    return reward
    

def simulated_bot_action_deterministic(
        balance,
        step_dict,
        action_array,
        next_exchange_price, # fed into neural net - part of state
        ticker,
        buy_now_reward,
        fee=0.001,
        allowance_fraction=1,
        base_asset_min=10,
        exchange_asset_min=10
    ):    
    '''
        e.g. balance = {'EUR': 500, 'BTC': 0.002, 'XRP': 100}
        
        Action_array:
            0 : buy 
            1 : sell
            2 : do nothing
        
    '''
    skip_next = False
    selected_action = torch.argmax(action_array, dim=1)
    
    exchange_asset, base_asset = ticker[:3], ticker[3:]
    reward = 0
    performed_action = 2
    
    # buy
    if selected_action[0] == 0:
        if balance[base_asset] >  base_asset_min:
            balance = simulated_market_buy(
                balance,
                ticker,
                1,
                next_exchange_price,
                fee,
                allowance_fraction
            )
            
            step_dict['base_balance_at_last_buy'] = get_base_balance(
                balance, next_exchange_price, ticker
            )
            reward = buy_now_reward
            performed_action = 0
            skip_next = True
            
        reward += get_negative_frequency_reward(step_dict)
       
    
    # sell
    elif selected_action[0] == 1:
        if balance[exchange_asset] >  exchange_asset_min:
            
            balance = simulated_market_sell(
                balance,
                ticker,
                1,
                next_exchange_price,
                fee,
                allowance_fraction
            )
            reward = reward_function(
                balance,
                next_exchange_price,
                ticker,
                step_dict['base_balance_at_last_buy']
            )
            performed_action = 1
            
            skip_next = True

        reward += get_negative_frequency_reward(step_dict)

    # do nothing
    elif selected_action[0] == 2:
        pass
    
    else:
        raise Exception('Error an action must be selected (Descrete), action array =', action_array)

    return balance, skip_next, reward, step_dict, performed_action


def get_base_balance(balance, next_exchange_price, ticker):
    """
    	Gets current balance in terms of the base asset
    """

    exchange_asset, base_asset = ticker[:3], ticker[3:]
    
    return balance[base_asset] + balance[exchange_asset]*next_exchange_price


def split_sim_into_parts(stream_date,
                         stream_time,
                         file_name_prefix='gui_episode_data',
                         file_path='data/streams/XRPEUR/',
                         max_hours_per_part=5,
                         include_mid_part=False):
    
    steps_per_part = int((3600*max_hours_per_part)/0.4)
    
    image_array = np.load(
        '{}{}_{}_images_{}_hours.npy'.format(
            file_path, file_name_prefix, stream_date, stream_time
        )
    )
    
    quote_array = np.load(
        '{}{}_{}_quotes_{}_hours.npy'.format(
            file_path, file_name_prefix, stream_date, stream_time
        )
    )
    
    parts = {'a': [0, steps_per_part], 'b': [steps_per_part, -1]}

    if include_mid_part:
        parts['c'] = [int(steps_per_part/2),
                      steps_per_part + int(steps_per_part/2)]
    
    for key, val in parts.items():
        print(val[0], val[1])
        print(quote_array[val[0]:val[1]].shape)
        np.save(
            '{}{}_{}{}_images_{}_hours.npy'.format(
                file_path, file_name_prefix, stream_date, key, str(max_hours_per_part)+'_0'
            ),
            image_array[val[0]:val[1]]
        )
        
        np.save(
            '{}{}_{}{}_quotes_{}_hours.npy'.format(
                file_path, file_name_prefix, stream_date, key, str(max_hours_per_part)+'_0'
            ),
            quote_array[val[0]:val[1]]
        )
