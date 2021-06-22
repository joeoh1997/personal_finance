# API Key : 6005cf417af8d2.86939217, f09ef0f6985bef8f53ad5f0ed68dc30c(financialmodellingprep.com)
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.linalg import inv
from matplotlib import pyplot

from .api_requests import get_price_data, get_statement_json

def static_ema(values, k):
    cur_ema = values[0]

    for val in values[1:]:
        cur_ema = val*k + cur_ema*(1 - k) 

    return cur_ema
	

def get_ema_growth(values, k):
    growths = []
	
    for i in range(len(values)-1):
        cur_growth = (values[i+1] - values[i])/values[i]
        growths.append(cur_growth)
		
    growth_ema = static_ema(growths, k)
    return growth_ema
	
	
def find_linear_forcast(
        values,
        X=None,
        xaxis='step',
        yaxis='values',
        title='',
        plot=False,
        reverse=False,
        forecast=False
    ):
    
    if reverse:
        values = list(values)
        values.reverse()
        
    if np.count_nonzero(values) == 0:
        return None, None, [0, 0]

    if X is None:
        size = len(values)+1
        X = np.ones([size, 2])
        X[:, 0] = np.arange(size)/size

    X_realized = X[:-1, :]

    max_val = np.max(np.abs(values))
    y = np.array(values)/max_val

	# linear least squares
    b = inv(X_realized.T.dot(X_realized)).dot(X_realized.T).dot(y)

    # predict using coefficients
    preds = X.dot(b) * max_val
	
    if plot:
		# plot data and predictions
        pyplot.scatter(X[:-1, 0], values)
        pyplot.plot(X[:, 0], preds, color='red')
        pyplot.xlabel(xaxis)
        pyplot.ylabel(yaxis)
        pyplot.title(title)
        pyplot.show()
	
    if forecast:
        forcast_value = int(preds[-1])
        if int(values[-1]) == 0:
            forcast_growth = None
        else:
            forcast_growth = round((forcast_value - values[-1])/values[-1], 4)
	
        return forcast_value, forcast_growth, b

    else:
        return b
	

def add_linear_forecast_params(
        input_data,
        results,
        varname,
        plot=False,
        reverse=True,
        add_forecast=False
    ):
    """
        Add linear regression forcast & weights for variable 'varname' to results dictionary
    """
    #print(varname)
    weights = find_linear_forcast(
        input_data, title=varname, plot=plot, reverse=reverse, forecast=add_forecast
    )

    if add_forecast:
        forcast_value, forcast_growth, weights = weights
        results['forcast_'+varname] = forcast_value
        results['pred_change_'+varname] = forcast_growth

    results['w0_'+varname] = weights[0]
    results['w1_'+varname] = weights[1]	
    
    return results


def get_price_data_between_date_range(
        ticker,
        start_date,
        forecast_date,
        end_date,
        price_data=None
    ):
    """
        Gets stock price data (e.g open, close) from between the specified date range.
    """
    start_date = to_date(start_date)
    
    if price_data is None:
        price_data = get_price_data(ticker)
        price_data['mid'] = price_data['low'] + (
            (price_data['high']-price_data['low'])/2
        )

    if forecast_date not in price_data['timestamp'].values:
        raise Exception('forecast date {} not in price data for {}, the company may be delisted'.format(forecast_date, ticker))

    price_data_range = price_data[price_data['timestamp'] >= start_date]    
    price_data_range = price_data_range[price_data_range['timestamp'] <= end_date]
    
    return price_data_range, price_data


def add_range_min_max_mid(
        model_data,
        price_data,
        varname,
        span,
        prefix='',
        affix=''
    ):
    """
        Adds min & max of price data aswel as final ema entry (mid)
    """
    model_data[str(prefix+'_min_'+varname+'_'+affix).replace('__', '_')] = price_data[varname].min()
    model_data[str(prefix+'_max_'+varname+'_'+affix).replace('__', '_')] = price_data[varname].max()
    
    ema = list(price_data[varname].ewm(span=span, adjust=False).mean())
    model_data[str(prefix+'_ema_last_'+varname+'_'+affix).replace('__', '_')] = ema[0]  
    
    return model_data, ema


def add_normal_and_ema_linear_models(
        model_data,
        price_data,
        varname,
        plot=False,
        span=20,
        ema=None,
        prefix='',
        affix='',
        include_ema_model=False
        
    ):
    """
        Adds linear model & ema linear model params to dictionary

    """
    save_name = str(prefix+'_'+varname+'_'+affix).replace('__', '_')
    
    model_data = add_linear_forecast_params(
        price_data[varname].values,
        model_data,
        save_name,
        plot=plot
    )
    
    if include_ema_model:
        if ema is None:
            ema = price_data[varname].ewm(span=span, adjust=False).mean()
        
        model_data = add_linear_forecast_params(
            ema,
            model_data,
            save_name+'_ema',
            plot=plot
        )
    
    return model_data
    

def add_stock_price_and_volume_linear_model_and_min_max_ema(
        ticker,
        model_data,
        start_date,
        forecast_date,
        end_date,
        span=20,
        price_type='mid',
        include_volume_models=False,
        include_ema_models=False,
        include_min_max=False,
        prefix='',
        plot=False,
    ):
    
    """
        Create Linear model over 'future/past' price &/ volume and add params to dict.
        Also add min max & ema final value of the price &/ volume.
    """
    
    start_date, forecast_date, end_date = [
        to_date(ddate) for ddate in [start_date, forecast_date, end_date]
    ]
    
    # get labels for forcast learning
    price_data, _ = get_price_data_between_date_range(
        ticker,
        start_date,
        forecast_date,
        end_date
    )
    
    if include_min_max:
        model_data, price_ema = add_range_min_max_mid(
            model_data,
            price_data,
            price_type,
            span,
            prefix=prefix,
            affix='price'
        )
    
    model_data = add_normal_and_ema_linear_models(
        model_data,
        price_data,
        price_type,
        plot,
        span,
        price_ema,
        prefix=prefix,
        affix='price',
        include_ema_model=include_ema_models
    )
    
    if include_volume_models:
        model_data = add_normal_and_ema_linear_models(
            model_data,
            price_data,
            'volume',
            plot,
            span,
            prefix=prefix,
            affix='',
            include_ema_model=include_ema_models
        )
        
        if include_min_max:
        
            model_data, price_ema = add_range_min_max_mid(
                model_data,
                price_data,
                'volume',
                span,
                prefix=prefix,
                affix=''
            )
    
    return model_data


def get_all_varnames(income_statements, cash_flow_statements):
    """
        gets all usable variables from income & cashflow statements
    """
    varnames = list(income_statements[0].keys()) + \
               list(cash_flow_statements[0].keys())
               
    varnames = [
        varname for varname in varnames if varname not in
        [
            'date', 'symbol', 'reportedCurrency', 'fillingDate',
            'acceptedDate', 'period', 'link', 'finalLink'
        ]
    ]

    return varnames
    
    
def get_statement_data(
        ticker,
        period,
        as_dataframe=False,
        merge_statements=False,
        min_statements=1,
    ):
    """
        Gets income & cash flow statement data for given ticker
    """
    if period not in ['quarter', 'year']:
        raise ValueError("period needs to be quarter or year...")
    
    income_statements = get_statement_json(ticker, period=period)
    
    cash_flow_statements = get_statement_json(
        ticker, period=period, statement_type='cash-flow-statement'
    )
    
    if not income_statements or not cash_flow_statements:
        raise Exception("No statements for {}".format(ticker))

    elif (len(income_statements) < min_statements 
            or len(cash_flow_statements) < min_statements):
        raise Exception(
            "Not enough statement data for {}, min={}, "
            "num income statements found={}, "
            "num cash flow statements found={}".format(
                ticker,
                min_statements,
                len(income_statements) if income_statements else None,
                len(cash_flow_statements) if cash_flow_statements else None
            )
        )

    if merge_statements:
        merged_statements = pd.DataFrame(income_statements).merge(
            pd.DataFrame(cash_flow_statements), on='date', how='outer', suffixes=['', '!!']
        )
        statements = merged_statements.drop(
            merged_statements.columns[merged_statements.columns.str.contains('!!')], axis=1
        )

    elif as_dataframe:
        statements = [pd.DataFrame(income_statements), pd.DataFrame(cash_flow_statements)]

    else:
        statements = income_statements, cash_flow_statements

    return statements


def get_statement_data_between_dates(
        ticker,
        period,
        start_date='2009-12-31',
        end_date=None,
        date_format='%Y-%m-%d',
        merge_statements=True,
        min_statements=1
    ):
    """
        Gets income & cash flow statement data for given ticker between specified dates
    """
    
    start_date = to_date(start_date)
    end_date = to_date(end_date)
    
    clipped_statements = []
    
    for statements in get_statement_data(ticker, period, True, merge_statements, min_statements):
        statements['date'] = pd.to_datetime(statements['date'], format=date_format).dt.date
        statements = statements[statements['date'] >= start_date]
        
        if end_date:
            statements = statements[statements['date'] <= end_date]
            
        clipped_statements.append(statements)
    
    return clipped_statements
    

def get_linear_model_data_from_statement(
        ticker,
        period,
        varnames='all',
        start_date='2009-12-31',
        end_date=None,
        min_statements=1,
        plot=False,
        model_data=None
    ):
    """ 
        Function makes linear models of each var e.g. revenue,
        either quarterly or yearly from the financial statements
    """
    
    model_data = {} if model_data is None else model_data
    dates, count = [], 0 

    start_date = to_date(start_date)
    end_date = to_date(end_date)
    
    income_statements, cash_flow_statements = get_statement_data(
        ticker, period, as_dataframe=False, min_statements=min_statements
    )

    if varnames == 'all':
        varnames = get_all_varnames(income_statements, cash_flow_statements)
    
    income_varnames = list(set(varnames).intersection(set(income_statements[0].keys()))) 

    for varname in varnames:
        data = []
        statements = income_statements if varname in income_varnames else cash_flow_statements
        
        for statement in statements:
            date = to_date(statement['date'])
            
            if count == 0:
                dates.append(date)
            
            if date >= start_date and ((date <= end_date) if end_date else True):
                data.append(statement[varname])

        if not data:
            raise(f'no {varname} data for {ticker}')

        data = np.array(data)
        data[data == None] = 0

        model_data = add_linear_forecast_params(
            data, model_data, varname+'_'+period, plot=plot, reverse=True
        )        
        count += 1

                        
    return model_data	


def to_date(date, date_format='%Y-%m-%d'):
    """
        Turns Datetime object or string to date object
    """
    try:
        if isinstance(date, str):
            date = datetime.strptime(date, date_format)
            
        if isinstance(date, datetime):
            date = date.date()
        
    except Exception:
        print('Enter string date in the specified'
              ' form {} or datetime/date object'.format(date_format))

    return date

	    
    
    



