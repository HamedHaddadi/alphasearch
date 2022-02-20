

import numpy as np
from math import factorial
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

class BinomialTree(metaclass=ABCMeta):
    """
    The base class for option pricing using binomial trees
    """
    def __init__(self, stock_price = None, strike_price = None,
             interest_rate = 0.04, maturity = 6, num_steps = 20, is_call = True, method = 'CRR', volatility=0.3, dividend = 0.0):
        """
        accepts:
        stock_price: initial stock price
        strike_price: option strike proce
        interest_rate: value of interest rate
        maturity: time to maturity
        num_steps: number of steps between 0 and maturity
        is_put: boolean; put or call
        method: CRR (Cox, Ross, Rubinstein)
        """
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.ir = interest_rate
        self._maturity = maturity
        self._num_steps = num_steps
        self.call_factor = {True:1, False:-1}[is_call]
        self.sigma = volatility
        self.div = dividend
        self.u = None
        self.d = None
        self.pu = None
        self.pd = None
        self.stock_tree = None
        self.binomial_probability = None
        self.method = method
        # compute up/down factors and probabilities
        self.method_map = {'CRR': self._compute_CRR}
        self.method_map[method]()
    
    def _fixed_div(self):
        return 0.0

    def _compute_CRR(self):
        """
        Cox, Ross and Rubinstein method
        """
        self.u = np.exp(self.sigma*np.sqrt(self.d_t))
        self.d = 1/self.u
        self.pu = (np.exp((self.ir - self.div)*self.d_t) - self.d)/(self.u - self.d)
        self.pd = 1 - self.pu

    @staticmethod
    def _non_zero(dates, prices):
        non_zero = {}
        for num in range(dates.shape[0]):
            date, price = dates[num], prices[:, num]
            non_zero[date] = price[price != 0][:, np.newaxis].T
        return non_zero

    @property
    def maturity(self):
        return self._maturity

    @maturity.setter
    def maturity(self, new_value):
        self._maturity = new_value;
        self.method_map[self.method]()

    @property
    def num_steps(self):
        return self._num_steps
    
    @num_steps.setter
    def num_steps(self, new_value):
        self._num_steps = new_value
        self.method_map[self.method]()
        
    @property
    def d_t(self):
        return self._maturity/self._num_steps
    
    @property
    def pricing_dates(self):
        return np.linspace(0, self._maturity, self.num_steps + 1, endpoint=True)

    # #### define all trees #### #
    # 1) stock price tree #
    def setup_stock_tree(self):
        self.stock_tree = np.zeros((self.num_steps + 1, self.num_steps + 1))
        for s in range(self.num_steps + 1):
            for i in range(s + 1):
                self.stock_tree[i,s] = self.stock_price*(self.u**i)*(self.d**(s - i))

    @abstractmethod
    def setup_payoff(self):
        ...

    @property
    @abstractmethod
    def option_price(self):
        ...  

    # ##### Various price representation ##### #
    @abstractmethod
    def show_price(self):
        ...
    
    def __call__(self):
        self.setup_stock_tree()
        self.setup_payoff()
        print('the option price array is = ', self.option_price)

          
# ######################################## #
#   Class for pricing American options     #
#                                          #
# ######################################## #

class AmericanOption(BinomialTree):

    def setup_payoff(self):
        self.payoff = np.zeros_like(self.stock_tree)
        for s in range(self.num_steps + 1):
            for i in range(s + 1):
                self.payoff[i][s] = np.max([0, self.call_factor*(self.stock_tree[i][s] - self.strike_price)])

    @property
    def option_price(self):
        self.setup_stock_tree()
        self.setup_payoff()
        price = np.zeros_like(self.payoff)
        for s in range(self.num_steps - 1, -1, -1):
            for i in range(s + 1):
                price[i][s] = (self.pu*self.payoff[i + 1][s + 1] + self.pd*self.payoff[i][s + 1])*np.exp(-1*(self.ir - self.div)*self.d_t)        
        return price
    
    def show_price(self):
        """
        displays stock price movemenets on possible exercise dates
        and corresponding option proces
        """
        stock_price = BinomialTree._non_zero(self.pricing_dates, self.stock_tree)
        option_price = BinomialTree._non_zero(self.pricing_dates, self.option_price)

        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs = axs.ravel()
        for date, price in stock_price.items():
            if price.size > 0:
                axs[0].plot(date, price, '.', ms = 2, mfc = 'black', mec = 'black')
        axs[0].set_xlabel('dates')
        axs[0].set_ylabel('stock price')

        for date, price in option_price.items():
            if price.size > 0:
                axs[1].plot(date, price, '.', ms = 2, mfc = 'blue', mec = 'blue')
        axs[1].set_xlabel('dates')
        axs[1].set_ylabel('stock price')
       

# ######################################## #
#   Class for pricing European options     #
#                                          #
# ######################################## #

class EuropeanOption(BinomialTree):

    def setup_binomial_probability(self):
        self.binomial_probability = np.zeros(self.num_steps + 1, self.num_steps + 1)
        for s in range(self.num_steps + 1):
            for i in range(s + 1):
                self.binomial_probability[i][s] = (factorial(s)/(factorial(s - i)*factorial(i)))*((self.pu**i)*(self.pd**(s - i)))    

