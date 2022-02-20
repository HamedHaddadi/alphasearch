

import pandas as pd 
from datetime import datetime, timedelta, date
from os import path 
from ..datautils.generate import IEXCloud, stock_history_from_iex, etf_price_history, get_sp500, stock_history_yahoo 
from datetime import date 
import sys

class MomentumMetrics:
    """
    class to compute momentum metrics for an asset or group of assets
    The best way to use the class is in a notebook. 
    """ 
    _wiki_price_points = ['open', 'high', 'low', 'close', 'adj_open', 'adj_high', 'adj_low', 'adj_close']

    def __init__(self, assets, tickers = None):
        """
        assets: data_frame of the assets, which include the closing price for a time period
        tickers: the specific ticker, or list of tickers
            if not specified, all tickers are used to calculate the metrics
        """
        self.assets = assets
        self.tickers= tickers
        self.benchmark = None
        
    def compute_momentum(self, lag_times=[1, 3, 6, 9, 12], start_time = None, sampling = 'monthly', plot = None):
        """
        computes the price momentum in monthly periods after the start_time
        lag_times are portfolio formation periods
        Jegadeesh & Titman (1993)
        I avoid using resample and pct_change to make sure dates are valid
        plot: None or a list of tickers
        """
        # find the closest available datetime to the time
        if start_time is None:
            start_time = self.assets.index.min()
        else:
            start_time = datetime.strptime(start_time, '%Y-%m-%d')
        start_idx = self.assets.index[self.assets.index.get_loc(start_time, method='nearest')]
        lag_factor = {'monthly': 30, 'daily': 1}[sampling]
        price_change = []
        date_index = []
        for lag in lag_times:
            lag_time = start_time + timedelta(days=lag*lag_factor)
            lag_idx = self.assets.index[self.assets.index.get_loc(lag_time, method='nearest')]
            if lag_idx == start_idx: continue 
            price_change.append((self.assets.loc[lag_idx, :].values - self.assets.loc[start_idx, :].values)*(1/self.assets.loc[start_idx, :].values))
            date_index.append(f'{lag_idx}')
        self.momentum = pd.DataFrame(price_change, columns = self.assets.columns)
        self.momentum.index = date_index
        self.momentum.dropna(axis = 1, how='all', inplace=True)
        self.momentum.index = pd.to_datetime(self.momentum.index)

    # calculation of RSI and associated methods and properties
    def compute_RSI(self, period = 14, date_range = None):
        """
        computes the relative strength index (RSI) in 14 day period intervals 
        if plot = None calculations are done for all tickers, then the RSI dataframe can be used for postprocessing
        if plot is specified as a list of tickers, then a plot for each ticker along with its price is generated
        """
        if date_range is None:
            date_range = (self.assets.index.min().strftime('%Y-%m-%d'),
                         self.assets.index.max().strftime('%Y-%m-%d'))
        assets = MomentumMetrics.choose_dates(self.assets, date_range)
        # screen the frame for start .. end range
        price_change = assets.pct_change()
        mean_gain = price_change[price_change > 0].fillna(0).rolling(period).mean().dropna()
        mean_loss =  price_change[price_change < 0].fillna(0).rolling(period).mean().abs().dropna()
        self.RSI = 100 - 100/(1 + mean_gain/mean_loss)

    def sp500_benchmark(self, start = '01-01-2010', end = date.today()):
        self.benchmark = get_sp500(start = start, end = end)
        
    # calculation of Bollinger bands frame
    @staticmethod
    def _bollinger_middle(row, period = 20):
        return row.rolling(period).mean()
    
    @staticmethod
    def _bollinger_upper(row, period = 20, k = 2):
        return row.rolling(period).mean() + k*row.rolling(period).std()
    
    @staticmethod
    def _bollinger_lower(row, period = 20, k = 2):
        return row.rolling(period).mean() - k*row.rolling(period).std()

    def compute_bollinger(self, period = 20, k = 2, date_range = ('2018-01-01', '2021-01-01')):
        
        assets = MomentumMetrics.choose_dates(self.assets, date_range)
        middle_df = assets.apply(self._bollinger_middle ,axis = 0, period = period).dropna()
        upper_df = assets.apply(self._bollinger_upper, axis = 0, period = period, k = k).dropna()
        lower_df = assets.apply(self._bollinger_lower, axis = 0, period = period, k = k).dropna()

        middle_df.columns = [col + '_middle' for col in assets.columns]
        upper_df.columns = [col + '_upper' for col in assets.columns]
        lower_df.columns = [col + '_lower' for col in assets.columns]
        self.bollinger = pd.concat([lower_df, middle_df, upper_df], axis = 1)

    @staticmethod
    def choose_dates(frame, date_range):
        start, end = tuple(date_range)
        if start:
            start = datetime.strptime(start, '%Y-%m-%d')
            frame = frame[frame.index.date >= start.date()]
        if end:
            end = datetime.strptime(end, '%Y-%m-%d')
            frame = frame[frame.index.date <= end.date()]
        return frame

    @property 
    def oversold(self):
        return self.RSI[self.RSI >= 70]
    
    @property
    def overbought(self):
        return self.RSI[self.RSI <= 30] 
    
    def ticker_valid(self, ticker):
        return ticker in self.momentum.columns 
    
    # factory methods for dataframe generation from various sources
    @classmethod
    def load_stocks_wiki(cls, file_info, file_format = 'csv', tickers = None, time = ['1900','2018'], price_point = 'close'):
        """
        generates a dataframe from quandl wiki files up until 2018. 
        files must be saved on the disk
        """
        if 'csv' in file_format:
            wiki_df = pd.read_csv(file_info, parse_dates = ['date'], index_col = ['date', 'ticker'], infer_datetime_format = True).sort_index()
        elif 'h5' in file_format:
            wiki_df = pd.read_hdf(file_info)
        idx = pd.IndexSlice
        
        if price_point not in cls._wiki_price_points:
            print('price points are not valid; choose a valid one from ', cls._wiki_price_points)
            sys.exit(-1)
        
        begin, end = tuple(time)
        if tickers == None:
            assets = wiki_df.loc[idx[begin:end, :], price_point].unstack('ticker')
        else:
            assets = wiki_df.loc[idx[begin:end, tickers], price_point].unstack('ticker')
        
        return cls(assets, tickers)

    @classmethod
    def load_stocks(cls, path_to_file = None):
        """
        loads stocks data from disk
        """
        assets = pd.read_csv(path_to_file, delimiter = ',', header = 0).apply(lambda col:
                                 pd.to_datetime(col) if col.name == 'Date' else col).set_index('Date')
        return cls(assets, assets.columns)

    
    @classmethod 
    def pull_stocks_iexcloud(cls, tickers = None, token = None, use_columns = 'close', save_path = None, 
     data_type = 'sandbox', history = '5y'):
        assets, ticker_names = stock_history_from_iex(tickers = tickers, token = token, data_type = data_type, 
                history = history, use_columns = use_columns, save_path = save_path) 
        return cls(assets, tickers = ticker_names)

    @classmethod 
    def pull_etf(cls, etfs = ['QQQ'], start = '01-01-2011', end = date.today()):
        assets = etf_price_history(etfs, date_range = (start, end))
        return cls(assets, tickers = etfs)
    
    @classmethod 
    def pull_stocks_yahoo(cls, tickers = None, history = '3y', use_columns = 'Close', save_path = None):
        assets = stock_history_yahoo(tickers = tickers, 
            history = history, use_columns = use_columns, save_path = save_path)         
        return cls(assets, tickers = tickers)
    



