
from . simulation import SimulatePortfolio 
from scipy.optimize import minimize 
from functools import partial, wraps 
import numpy as np
import sys 

class MeanVariance(SimulatePortfolio):
    """
    generates optimum weights based on various constraints
    """
    def __init__(self, frame, **kwargs):
        super().__init__(frame)
        self.num_assets= len(self.asset_frame.columns)
        self.w0 = np.random.uniform(0, 1, self.num_assets)
        self.w0 /= np.sum(self.w0)
    
    def opt_counter(func):
        @wraps(func)
        def opt_wrapper(self, date_range = None, sampling = 'M',
                 method = 'SLSQP', renew = False, **kwargs):
            opt_wrapper.counter += 1
            if (self.period_returns is None and self.risk_free is None) or renew:
                if date_range is not None:
                    start, end = MeanVariance.time_range(date_range)
                    self.generate_period_returns(date_range = (start, end), sampling = sampling)
                    self.risk_free = MeanVariance.interest_rates(sampling, start, end).mean().squeeze()
                else:
                    print('define a date range')
                    exit(0)
            return func(self, date_range = date_range, sampling = sampling, method = method, **kwargs)
        opt_wrapper.counter = 0
        return opt_wrapper

    # ### Optimize based on max sharpe ratio ### #
    @opt_counter     
    def optimize_max_sharpe(self, date_range = None, sampling = 'M', method = 'SLSQP', renew=False):
        print('performing optimization for max sharpe ratio for ', self.optimize_max_sharpe.counter, ' time ...')
    
        bounds = [[0,1]]*self.num_assets 
        if self.sampling:
            sampling = self.sampling
        constraints = {'type': 'eq', 
                            'fun': lambda x: np.sum(np.abs(x)) - 1}

        return minimize(MeanVariance.neg_sharpe, self.w0, args = (self.period_returns,
                     self.risk_free, sampling), 
                                method = method, 
                                    bounds = bounds, 
                                        constraints = constraints, 
                                           options = {'maxiter': 1e3})

    # function to be minimized 
    @staticmethod 
    def neg_sharpe(weights, returns, mean_interest = 0, sampling = 'M'):
        sharpe = MeanVariance.portfolio_sharpe(weights, returns,
                                mean_interest = mean_interest, sampling = sampling)
        return -1*sharpe 

    # ###  Optimize based on minimum volatility ### #
    @opt_counter
    def optimize_min_volatility(self, date_range = None, sampling = 'M', method = 'SLSQP', renew = False):
        print('performing optimization for minimum volatility for ', self.optimize_min_volatility.counter, ' time ...')
        
        bounds = [[0,1]]*self.num_assets 
        
        if self.sampling:
            sampling = self.sampling

        constraints = {'type': 'eq', 
                            'fun': lambda x: np.sum(np.abs(x)) - 1} 
         
        return minimize(MeanVariance.portfolio_volatility, self.w0, args = (self.period_returns, sampling), 
                                method = method, 
                                    bounds = bounds, 
                                        constraints = constraints, 
                                           options = {'maxiter': 1e3})

    # ### Optimize based on maximum volatility ### #
    @opt_counter
    def optimize_max_return(self, date_range = None, sampling = 'M', method = 'SLSQP', renew = False):
        print('performing optimization for maximum volatility for ', self.optimize_max_return.counter, ' time ...')

        bounds = [[0,1]]*self.num_assets 

        if self.sampling:
            sampling = self.sampling 
        
        constraints = {'type': 'eq', 
                            'fun': lambda x: np.sum(np.abs(x)) - 1}
        
        return minimize(MeanVariance.neg_return, self.w0, args = (self.period_returns, sampling), 
                            method = method, 
                                bounds = bounds, 
                                    constraints = constraints, 
                                        options = {'maxiter': 1e3})

    @staticmethod 
    def neg_return(weights, returns, sampling = 'M'):
        returns = MeanVariance.portfolio_return(weights, returns, sampling = sampling)
        return -1*returns 


         



    


        




    








     
