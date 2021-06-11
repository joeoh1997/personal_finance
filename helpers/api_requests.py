# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:16:46 2021

@author: Joe
"""
from datetime import datetime
import requests
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup


def request(url):
    response = requests.get(url)
    return response.json()


def request_html_as_mozilla(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    return webpage.decode("utf-8")


def financial_model_prep_growth(ticker, limit=1, period='year', api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'):
    return request(
        f"https://financialmodelingprep.com/api/v3/financial-growth/{ticker}?limit={limit}&period={period}&apikey={api_key}"
    )


def financial_model_prep_quote(ticker='BTCUSD',
                               api_key='f09ef0f6985bef8f53ad5f0ed68dc30c',
                               drop_columns=False):
    if drop_columns:
        if type(drop_columns) == bool:
            drop_columns = ["symbol", "name", "marketCap",
                            "exchange", "eps", "pe",
                            "earningsAnnouncement",
                            "sharesOutstanding",
                            "timestamp"]
            
    range_varnames = ['dayLow', 'dayHigh', 'yearHigh', 'yearLow']
    
    tickers = [ticker] if type(ticker) == str else ticker
    
    url = f"https://financialmodelingprep.com/api/v3/quote/{','.join(tickers)}?apikey={api_key}"
    
    return request(url), drop_columns, range_varnames
    
    
def binance_quote(ticker='BTCEUR', drop_columns=False):
    
    if drop_columns:
        if type(drop_columns) == bool:
            drop_columns = ["symbol", "openTime", "closeTime", "firstId", "lastId"]
        
    range_varnames = ['highPrice', 'lowPrice']
    
    tickers = [ticker] if type(ticker) == str else ticker
    
    quotes = []
    for ticker in tickers:
        got_data = False
        
        while not got_data:
            try:
                quotes.append(
                    request(
                        f"https://api.binance.com/api/v3/ticker/24hr?symbol={ticker}"
                    ) 
                )
                got_data = True
            except Exception:
                print("binance exception, retrying...")
              
    return quotes, drop_columns, range_varnames


def get_qoute(ticker='BTCEUR', add_differences=False, binance=False, drop_columns=False):
    
    if binance:
        quotes, drop_columns, range_varnames =\
            binance_quote(ticker, drop_columns)
            
    else:
        quotes, drop_columns, range_varnames =\
            financial_model_prep_quote(
                ticker, drop_columns=drop_columns
            )
 
    mod_qoutes = []
    
    for quote in quotes:

        if add_differences:
            for varname in range_varnames:
                diff_var_name = varname+'_diff'
                quote[diff_var_name] = float(quote['lastPrice' if binance else 'price']) - float(quote[varname])    
                
        if drop_columns:
            quote = {key: value for key, value in quote.items() if key not in drop_columns}
        mod_qoutes.append(quote)
        
    return mod_qoutes if len(mod_qoutes) > 1 else mod_qoutes[0]


def get_financial_ratios(ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'):
    url =  f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={api_key}"
    response = requests.get(url)
    return response.json()[0]

   
def get_statement_json(ticker, limit=120, period='yearly', statement_type='income-statement', api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'):
    url = f'https://financialmodelingprep.com/api/v3/{statement_type}/{ticker}?limit={limit}&apikey={api_key}&period={period}'
    response = requests.get(url)
    return response.json()


def get_num_shares(ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'):
    price = requests.get(
        f'https://financialmodelingprep.com/api/v3/quote-short/{ticker}?apikey={api_key}'
        ).json()[0]['price']
    
    market_cap = requests.get(
        f'https://financialmodelingprep.com/api/v3/market-capitalization/{ticker}?apikey={api_key}'
    ).json()[0]['marketCap']
    
    return int(market_cap/price)


def get_tickers(query, exchange='NASDAQ', limit=10, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'):
    url = f'https://financialmodelingprep.com/api/v3/search-ticker?query={query}&limit={limit}&exchange={exchange}&apikey={api_key}'
    response = requests.get(url)
    return response.json()


def get_all_tickers_by_exchange(exchange='NASDAQ', api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'):
    alphabet = [chr(i) for i in range(65, 91)]
    tickers = []
    
    for letter in alphabet:
        tickers.extend(pd.DataFrame(
            get_tickers(letter, exchange=exchange, limit=10000, api_key=api_key)
        )['symbol'].values)

        tickers = list(set(tickers))
        
    return tickers


def get_all_tradeable_tickers(
        include_price_and_info=False, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
    ):
    url = f'https://financialmodelingprep.com/api/v3/available-traded/list?apikey={api_key}'
    tickers = requests.get(url).json()

    if not include_price_and_info:
        tickers = pd.DataFrame(tickers)['symbol'].values 
    
    return tickers

def get_price_data(ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'):
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={api_key}'
    response = requests.get(url)
    price_data = pd.DataFrame()
    
    for line in response.json()['historical']:
        price_data = price_data.append(line, ignore_index=True)
    
    price_data['timestamp'] = pd.to_datetime(price_data['date'], format="%Y-%m-%d").dt.date
    return price_data


def get_minute_price_data(ticker,
                          start_readtime=None,
                          end_readtime=None,
                          api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'):
    data = request(
        f'https://financialmodelingprep.com/api/v3/historical-chart/1min/{ticker}?apikey={api_key}'
    )
    
    if start_readtime:
        mod_data = []
        index = 1
        
        while(index):
            minute_data = data[index-1]
            index += 1
            
            quote_datetime = datetime.strptime(
                minute_data['date'], "%Y-%m-%d %H:%M:%S"
            ) 
            
            if quote_datetime <= end_readtime:
                if quote_datetime >= start_readtime:
                    mod_data.append(minute_data)
                else:
                    index = False
        
        data = mod_data
    return data

    
def get_premarket_movers(num_movers=10):
    html = request_html_as_mozilla(
        'https://www.benzinga.com/premarket/'
    )
    
    soup = BeautifulSoup(html, "lxml")
    mover_dict = {}
    
    headers = ['name', 'price', 'change', 'volume']
    
    for table in soup.find_all('table'):
        if 'Stock' in str(table.find('th')):
            
            for tr in table.find_all("tr")[:int(num_movers/2)+1]:
                tds = tr.find_all("td")
                
                if len(tds) > 0:
                    ticker = tds[0].text.strip()
                    ticker_dict = {
                        headers[i]: tds[i+1].text.strip().replace('\n', '') for i in range(4)
                    }
                    
                    ticker_dict['price'] = ticker_dict['price'].replace('$','')
                    
                    if 'M' in ticker_dict['volume']:
                        ticker_dict['volume'] = float(ticker_dict['volume'].strip('M')) * 1000000
                    elif 'K' in ticker_dict['volume']:
                        ticker_dict['volume'] = float(ticker_dict['volume'].strip('K')) * 1000
                    else:
                        ticker_dict['volume'] = float(ticker_dict['volume'])
                        
                    mover_dict[ticker] = ticker_dict
                    
    return mover_dict