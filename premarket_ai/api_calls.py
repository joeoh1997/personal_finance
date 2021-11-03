import time
import os
import requests
import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

from helpers.api_requests import request_html_as_mozilla
from helpers.linear_forecasts import find_linear_forcast


def request(url):
    print(url)
    response = requests.get(url)
    return response.json()


def company_profile(
    ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):

    return pd.Series(request(
        "https://financialmodelingprep.com/api/v3/profile/"
        f"{ticker}?apikey={api_key}"
    )[0])


def discounted_cash_flow(
    ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    return pd.Series({
        'dcf': request(
            "https://financialmodelingprep.com/api/v3/"
            f"discounted-cash-flow/{ticker}?apikey={api_key}"
        )[0]['dcf']
    })


def shares_float(
    ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    return pd.Series(request(
        "https://financialmodelingprep.com/api/v4/shares_float?"
        f"symbol={ticker}&apikey={api_key}"
    )[0]).drop(['source', 'date'])
    


def social_sentiment(
    ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    social_sentiment_ = request(
        "https://financialmodelingprep.com/api/v4/social-sentiment?"
        f"symbol={ticker}&limit=1&apikey={api_key}"
    )[0]

    return pd.Series(social_sentiment_)


def key_metrics_ttm(
    ticker, 
    api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    return pd.Series(request(
        "https://financialmodelingprep.com/api/v3/key-metrics-ttm/"
        f"{ticker}?apikey={api_key}&limit=1"
    )[0]).drop(['priceToSalesRatioTTM', 'peRatioTTM'])


def ratios_ttm(
    ticker, 
    api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    return pd.Series(request(
        "https://financialmodelingprep.com/api/v3/ratios-ttm/"
        f"{ticker}?apikey={api_key}"
    )[0])


def insider_trading(
    ticker, limit=10, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    """Not used"""
    return request(
        "https://financialmodelingprep.com/api/v4/insider-trading?"
        f"symbol={ticker}&limit={limit}&apikey={api_key}"
    )


def analyst_estimates(
    ticker,
    api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    estimate = pd.Series(request(
        "https://financialmodelingprep.com/api/v3/analyst-estimates/"
        f"{ticker}?period=quarter&limit=1&apikey={api_key}"
    )[0])
    estimate['analyst_estimate_forecast_date'] = estimate['date']

    return estimate.drop('date')


def financial_statement_trends(
    ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    def find_trends(data, varname, trend_dict):
        weights = find_linear_forcast(
            data[varname].astype(np.float), title=varname, plot=False
        )

        trend_dict[varname+'trend_slope'] = weights[0]
        trend_dict[varname+'trend_constant'] = weights[1]

    statement_data = pd.DataFrame(request(
        "https://financialmodelingprep.com/api/v3/"
        f"financial-statement-full-as-reported/{ticker}?apikey={api_key}"
    ))

    statement_data['date'] = pd.to_datetime(
        statement_data['date'], format="%Y-%m-%d"
    )
    statement_data = statement_data.sort_values(by='date')[-5:]
    statement_data["long_term_debt"] = statement_data["longtermdebtcurrent"] +\
         statement_data["longtermdebtnoncurrent"]

    trend_dict = {}

    find_trends(statement_data, "long_term_debt", trend_dict)
    find_trends(statement_data, "grossprofit", trend_dict)
    find_trends(statement_data, "netincomeloss", trend_dict)
    find_trends(statement_data, "stockholdersequity", trend_dict)

    return pd.Series(trend_dict)


def quarterly_earnings(
    ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    earnings =  request(
        "https://financialmodelingprep.com/api/v3/historical/"
        f"earning_calendar/{ticker}?limit=10&apikey={api_key}"
    )

    last_real_earnings = None
    
    for earning in earnings:
        earning = pd.Series(earning)

        if earning['eps']:
            last_real_earnings = earning.add_prefix('quarterly_')
            break

    return last_real_earnings


def earning_surprise_trend(ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'):

    earnings =  pd.DataFrame(request(
        "https://financialmodelingprep.com/api/v3/"
        f"earnings-surprises/{ticker}?limit={8}&apikey={api_key}"
    ))
    earning_data = {} 

    earnings['date'] = pd.to_datetime(
        earnings['date'], format="%Y-%m-%d"
    )
    earnings = earnings.sort_values(by='date')

    earnings['surprises'] = earnings['actualEarningResult'] - earnings['estimatedEarning']

    earning_data["latest_earinings"] = earnings.tail(1).values
    earning_data["previous_earinings"] = earnings.tail(2).values

    earning_data["latest_earining_surprise"] = earnings.tail(1)['surprises']
    earning_data["previous_earining_surprise"] = earnings.tail(2)['surprises']

    weights = find_linear_forcast(
        earnings['actualEarningResult'], title='actualEarningResult', plot=False
    )

    earning_data['earnings_trend_slope'] = weights[0]
    earning_data['earnings_trend_constant'] = weights[1]

    weights = find_linear_forcast(
        earnings['surprises'], title='surprises', plot=False
    )

    earning_data['earnings_surprises_trend_slope'] = weights[0]
    earning_data['earnings_surprises_trend_constant'] = weights[1]

    return pd.Series(earning_data)


def price_gradings(
    ticker, limit=100, format="%Y-%m-%d", api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    grades = {
        "Equal-Weight":0, "Buy":0, "Neutral":0,
        "Sell":0, "Overweight":0, "Outperform":0,
        "Hold":0, "Underperform":0, "Market Perform":0
    }  

    def get_latest_grade(group):
        return group.sort_values(by='date').tail(1)

    start_date = datetime.today() - relativedelta(months=6)

    gradings = pd.DataFrame(request(
        "https://financialmodelingprep.com/api/v3/grade/"
        f"{ticker}?limit={limit}&apikey={api_key}"
    ))

    gradings['date'] = pd.to_datetime(
        gradings['date'], format=format
    )

    gradings = gradings[gradings['date'] > start_date].groupby(
        by='gradingCompany'
    ).apply(get_latest_grade)


    for var, val in gradings['newGrade'].value_counts().iteritems():
        grades[var] = val

    return pd.Series(grades)


def stock_news(
    ticker, limit=20, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    return pd.Series(request(
        "https://financialmodelingprep.com/api/v3/stock_news?"
        f"tickers={ticker},?limit={limit}&apikey={api_key}"   ##### FIX !!!!!!!!!!!! ~~~~~~
    )[0]).add_prefix('news_')


def sector_price_earning_ratio(
    exchange, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    return sector_industry_price_earning_ratio(
        exchange, 'sector', api_key
    )
 

def industry_price_earning_ratio(
    exchange, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    return sector_industry_price_earning_ratio(
        exchange, 'industry', api_key
    )


def sector_industry_price_earning_ratio(
    exchange, query_type='sector', api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    today = date.today().strftime("%Y-%m-%d")

    ratio_data = request(
        f"https://financialmodelingprep.com/api/v4/{query_type}_price_earning_ratio?"
        f"date={today}&exchange={exchange}&apikey={api_key}"
    )
    return pd.Series({
        f"pe_{ratio_data_[query_type]}_{ratio_data_['exchange']}": ratio_data_['pe']
        for ratio_data_ in ratio_data
    })


def sectors_performance(
    api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    changes =  request(
        f"https://financialmodelingprep.com/api/v3/sectors-performance?apikey={api_key}"
    )

    return pd.Series({
        f"changesPercentage_{sector['sector']}": sector['changesPercentage']
        for sector in changes
    })


def income_statement_growth(
    ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    return pd.DataFrame(request(
        "https://financialmodelingprep.com/api/v3/income-statement-growth/"
        f"{ticker}?limit=4&apikey={api_key}&period=quarter"
    )).drop(['date', 'period'], axis=1).mean()


def last_year_metrics(
    ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    varnames = [
        "revenues", "longtermdebtnoncurrent", "longtermdebtcurrent", "netincomeloss",
        "researchanddevelopmentexpense", "sellinggeneralandadministrativeexpense",
        "nonoperatingincomeexpense", "operatingexpenses", 
        "cashcashequivalentsrestrictedcashandrestrictedcashequivalents",
        "netcashprovidedbyusedinfinancingactivities",
        "netcashprovidedbyusedininvestingactivities",
        "netcashprovidedbyusedinoperatingactivities",
        "assets", "liabilities",
    ]

    last_year_metrics_ = pd.DataFrame(request(
        "https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/"
        f"{ticker}?period=year&limit=4&apikey={api_key}&period=quarter"
    ))[varnames].mean()

    return last_year_metrics_
    

def five_min_price_volume(
    ticker, limit=10, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    """Not used"""
    return get_price_volume(
        "historical-chart/5min", ticker, limit, api_key
    )


def one_hour_price_volume(
    ticker, limit=10, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    """Not used"""
    return get_price_volume(
        "historical-chart/1hour", ticker, limit, api_key
    )


def four_hour_price_volume(
    ticker, limit=10, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    """Not used"""
    return get_price_volume(
        "historical-chart/4hour", ticker, limit, api_key
    )


def get_price_volume(
    price_type, ticker, limit=10, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    """Not used"""
    res =  request(
        f"https://financialmodelingprep.com/api/v3/"
        f"{price_type}/{ticker}?apikey={api_key}"
    )[:limit]

    return pd.DataFrame(res).iloc[::-1]


def one_day_price_volume(
    ticker, limit=10, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c'
):
    res =  request(
        "https://financialmodelingprep.com/api/v3/"
        f"historical-price-full/{ticker}?apikey={api_key}"
    )['historical'][:limit]

    return pd.DataFrame(res).iloc[::-1]


def historic_price_data(ticker, api_key='f09ef0f6985bef8f53ad5f0ed68dc30c', prefix=''):
    def ema_latest_value(data, varname, span):
        return float(data[varname].ewm(span=span).mean().tail(1))

    historic_data = {}
    data = one_day_price_volume(ticker, 252, api_key)

    data['date'] = pd.to_datetime(
        data['date'], format="%Y-%m-%d"
    )

    data = data.sort_values(by='date')

    weights = find_linear_forcast(
        data['close'], title='close_price', plot=False
    )

    historic_data[prefix+'one_year_price_trend_slope'] = weights[0]
    historic_data[prefix+'one_year_price_trend_constant'] = weights[1]

    weights = find_linear_forcast(
        data.iloc[192:]['close'], title='close_price', plot=False
    )

    historic_data[prefix+'three_month_price_trend_slope'] = weights[0]
    historic_data[prefix+'three_month_price_trend_constant'] = weights[1]

    historic_data[prefix+'close_price_200_day_ema'] =\
        ema_latest_value(data, 'close', 200)

    historic_data[prefix+'close_price_100_day_ema'] =\
        ema_latest_value(data, 'close', 100)

    historic_data[prefix+'close_price_50_day_ema'] =\
        ema_latest_value(data, 'close', 50)

    historic_data[prefix+'changePercent_50_day_ema'] =\
        ema_latest_value(data, 'changePercent', 50)

    historic_data[prefix+'changePercent_200_day_ema'] =\
        ema_latest_value(data, 'changePercent', 200)

    historic_data[prefix+'one_year_changePercent_std'] = data['changePercent'].std()
    historic_data[prefix+'three_month_changePercent_std'] = data.iloc[192:]['changePercent'].std()

    yesterday = data.tail(1)
    
    if 'volume' in data.columns:
        historic_data[prefix+'one_year_volume_std'] = data['volume'].std()
        historic_data[prefix+'three_month_volume_std'] = data.iloc[192:]['volume'].std()

        historic_data[prefix+'one_year_volume_mean'] = data['volume'].mean()
        historic_data[prefix+'three_month_volume_mean'] = data.iloc[192:]['volume'].mean()

        historic_data[prefix+'vwap'] = float(yesterday['vwap'])#data['vwap'].tail(1)

    historic_data[prefix+'yesterday_close'] = float(yesterday['close'])
    historic_data[prefix+'yesterday_open'] = float(yesterday['open'])
    historic_data[prefix+'yesterday_high'] = float(yesterday['high'])
    historic_data[prefix+'yesterday_low'] = float(yesterday['low']) #2021-10-04T00:00:00.000000000
    historic_data[prefix+'yesterday_day'] = yesterday['label'].values[0]
    historic_data[prefix+'yesterday_date'] = \
        pd.to_datetime(yesterday['date'].values[0], format="%Y-%m-%d").strftime("%Y-%m-%d")

    historic_data[prefix+'one_month_high'] = data.iloc[232:]['high'].max()
    historic_data[prefix+'one_month_low'] = data.iloc[232:]['low'].min()

    historic_data[prefix+'three_month_high'] = data.iloc[192:]['high'].max()
    historic_data[prefix+'three_month_low'] = data.iloc[192:]['low'].min()

    historic_data[prefix+'one_year_high'] = data['high'].max()
    historic_data[prefix+'one_year_low'] = data['low'].min()

    return pd.Series(historic_data)


def fear_and_greed():
    url = "https://fear-and-greed-index.p.rapidapi.com/v1/fgi"

    headers = {
        'x-rapidapi-host': "fear-and-greed-index.p.rapidapi.com",
        'x-rapidapi-key': "8be8c38ba4mshae21598071a1816p131389jsnaa1c9987e9d8"
        }

    response = requests\
        .request("GET", url, headers=headers).json()['fgi']

    return pd.Series({
        'fear_and_greed_today': response['now']['value'],
        'fear_and_greed_yesterday': response['previousClose']['value']
    })


def get_premarket_price(ticker, save_html=False):
    html = request_html_as_mozilla(f"https://www.benzinga.com/quote/{ticker}")
    if save_html:
        with open('html.txt', 'w') as f:
            f.write(html) 
    return html.split('hidden md:table-cell text-right')[3].split('<')[0].split('>')[1]



def pipeline(tickers, path):


    non_ticker_functions = {
        sector_price_earning_ratio: [
            {"exchange":"nasdaq"}, {"exchange":"NSYE"}, {"exchange":"EURONEXT"}, {"exchange":"LSE"}
        ], 
        industry_price_earning_ratio: [
            {"exchange":"nasdaq"}, {"exchange":"NSYE"}, {"exchange":"EURONEXT"}, {"exchange":"LSE"}
        ], 
        sectors_performance: [{}], 
        fear_and_greed: [{}]
    }

    per_ticker_functions = [
        company_profile, shares_float, #discounted_cash_flow - in some other call
        social_sentiment, key_metrics_ttm,
        ratios_ttm, analyst_estimates, quarterly_earnings,
        earning_surprise_trend, price_gradings, stock_news,
        income_statement_growth, last_year_metrics, 
        historic_price_data, financial_statement_trends
    ]

    data = None

    for ticker in tickers:
        for func in per_ticker_functions:
            try:
                if data is None:
                    data = func(ticker)
                else:
                    data = pd.concat([data, func(ticker)])
                print(f'Added data for function : {func}\n')

            except Exception as e:
                print(
                    f'Failed adding data for function : {func}\n'
                    f'Error message : {e}'
                )
        data = data.drop_duplicates()
        indxs = []
        for ind in data.index:
            if ind in indxs:
                print(ind)
                print(data[ind])
                print("\n")
            else:
                indxs.append(ind)
        data.to_json(f"{path}{ticker}.json")

        data = None

        for func, vars in non_ticker_functions.items():
            for vars_ in vars:
                try:
                    if data is None:
                        data = func(**vars_)
                    else:
                        data = pd.concat([data, func(**vars_)])
                    print(f'Added data ::\n\tFunction : {func}\n\tParams : {vars_}\n')

                except Exception as e:
                    print(
                        f'Failed ::\n\tFunction : {func}\nParams : {vars_}\n'
                        f'\tError message : {e}'
                    )

        data.drop_duplicates().to_json(f"{path}market_state.json")


    market_tickers = [
        "^FVX","^TNX", "^TYX",
        "^VIX", "^OVX", "^GVZ",
        "BTCUSD", "EURUSD", 
        "GCUSD", "CLUSD",
        "^GSPC", "^DJI", "^IXIC"
    ]

    data = None

    for ticker in market_tickers:
        try:
            if data is None:
                data = historic_price_data(ticker, prefix=f"{ticker}_")
            else:
                data = pd.concat([
                    data, 
                    historic_price_data(ticker, prefix=f"{ticker}_")
                ])
            print(f'Added historic data for : {ticker}\n')

        except Exception as e:
            print(
                f'Failed adding data for : {ticker}\n'
                f'Error message : {e}'
            )

    data.to_json(f"{path}_market_outlook.json")


    todays_date = date.today()
    read_time = datetime(
        todays_date.year,
        todays_date.month,
        todays_date.day,
        13,
        30,
        00
    )
        
    print('Started, waiting until', read_time.strftime("%H:%M:%S"))
    
    while(datetime.now() <= read_time):
        pass

    premarket_prices = []

    for _ in range(19): 
        prices = {'time': datetime.now()}
        for ticker in tickers+['SPY', 'GLD', 'DIA', 'QQQ', 'TLT']:
            prices[ticker] = get_premarket_price(ticker)

        premarket_prices.append(prices)
        print("\nSleeping for 5 minutes..\n")
        time.sleep(300)

    pd.DataFrame(premarket_prices).to_csv(f"{path}_premarket_prices.csv")


if __name__ == '__main__':
    pipeline(['AAPL', 'BABA'])



