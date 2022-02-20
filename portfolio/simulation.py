
from datetime import datetime 
import numpy as np
import pandas as pd 
from .. datautils.generate import stock_history_from_iex, interest_rates
import sys

# ##### Simulate Portfolios ##### #
# Base class for computing effective frontier
# ############################### #

class SimulatePortfolio:
    """
    main portfolio simulation class
    It uses a dataframe of stock history and determines the
        weights; 
    Based on weight, this class computes Sharpe ratio,
        return and volatility  (All annualized)
    """
    _annualize = {'M': 12, 
                    'D': 252, 
                        'W': 52}

    def __init__(self, asset_frame, tickers = None, **kwargs):
        self.asset_frame = asset_frame 
        self.sampling = kwargs.get('sampling')
        self.period_returns = None
        self.risk_free = None 
   
    # ##### All methods to simulate portfolios ##### #
    # pull interest rates
    time_range = staticmethod(lambda date_range:  tuple([datetime.strptime(elem, '%Y-%m-%d') for elem in date_range]))    

    def generate_period_returns(self, date_range = None, sampling = 'M'):
        if date_range is None:
            start, end = SimulatePortfolio.time_range(date_range)
        else:
            start, end = date_range 
        asset_frame = self.asset_frame[(self.asset_frame.index.date >= start.date()) &
                             (self.asset_frame.index.date <= end.date())]
        if asset_frame.isnull().sum().sum() > 0:
            print('warning! there is NaN values in the dataframe ...')
            print('forward fill is done; but excpect loss in accuracy ')
            asset_frame = asset_frame.fillna(method='ffill').dropna(axis = 1)
    
        self.period_returns = asset_frame.resample(sampling).last().pct_change().dropna()        

    # prepare data and simulate portfolios for multiple
    # randomly generated portfolios
    def simulate_random_portfolios(self, date_range = None, sampling = 'M', num_portfolios = 100, short = False):
        if self.sampling is not None:
            sampling = self.sampling 
        
        start, end = SimulatePortfolio.time_range(date_range)
        self.generate_period_returns(date_range = (start, end), sampling = sampling)
        self.risk_free = interest_rates(sampling, start, end).mean().squeeze()
        _,num_tickers = self.period_returns.shape 
        alpha = np.full(shape = num_tickers, fill_value = 0.05)
        weights = np.random.dirichlet(alpha = alpha, size = num_portfolios)
        if short:
            weights *= np.random.randint(-1, 2, weights.shape)
        mean_returns = np.matmul(weights, self.period_returns.T).mean(1)
        #volatility = np.sqrt(np.matmul(weights, np.matmul(self.period_returns.cov(), weights.T)))
        volatility = np.matmul(weights, self.period_returns.T).std(1)
        sharpe = ((mean_returns - self.risk_free)/volatility)*np.sqrt(SimulatePortfolio._annualize[sampling])

        # Note:
        #   I defined weight as an attribute; to find indeces of the best portfolios
        self.random_portfolios = pd.DataFrame(np.c_[weights, mean_returns, volatility, sharpe],
             columns = self.asset_frame.columns.tolist() + ['returns', 'volatility', 'sharpe'])
        return sharpe, mean_returns, volatility   

    # other staticmethods useful for portfolio optimization
    # weights is a one dimensional arrary: (n_assets,)
    @staticmethod 
    def portfolio_volatility(weights, returns, sampling = 'M'):
        return np.sqrt(np.matmul(weights[:, np.newaxis].T,np.matmul(returns.cov(), weights))* SimulatePortfolio._annualize[sampling]) 
        
    
    @staticmethod 
    def portfolio_return(weights, returns, sampling = 'M'):
        return np.matmul(returns, weights).mean()*SimulatePortfolio._annualize[sampling]
    

    @staticmethod 
    def portfolio_sharpe(weights, period_returns, mean_interest = 0,  sampling = 'M'):
        mean_returns = SimulatePortfolio.portfolio_return(weights, period_returns, sampling = sampling)
        volatility = SimulatePortfolio.portfolio_volatility(weights, period_returns, sampling = sampling)
        return (mean_returns - mean_interest)/volatility 
    
    # ##### methods to generate data from various sources ##### #
    @classmethod 
    def pull_from_iexcloud(cls, tickers = None, token = None,
                 use_columns = 'close', save_path = None,
                  data_type = 'sandbox', history = '5y', **kwargs):
        
        asset_frame, ticker_names = stock_history_from_iex(tickers = tickers, token = token, data_type = data_type, 
                    history = history, use_columns = use_columns, save_path = save_path) 
        
        return cls(asset_frame, tickers = ticker_names, **kwargs)
        