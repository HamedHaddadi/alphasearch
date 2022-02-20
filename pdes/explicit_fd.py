

from abc import ABCMeta, abstractmethod
import numpy as np 
from functools import wraps

class ExplicitAdvectionDiffusion(metaclass = ABCMeta):
    """
    Transient Advection-Diffusion Base class 
    """

    def __init__(self, asset_limits = None, strike_price = None, maturity = None,
             asset_resolution = None, time_resolution = None, interest_rate = 0, volatility = None, alpha = None, beta=None):

        self.asset_low, self.asset_high = asset_limits
        self.strike_price = strike_price
        self.maturity = maturity
        self.asset_resolution = asset_resolution
        self.time_resolution = time_resolution
        self.asset_price_range = np.linspace(self.asset_low, self.asset_high, self.asset_resolution)
        self.r = interest_rate
        self.sigma = volatility
        self._alpha = {False: 1, True:alpha}[alpha != None]
        self._beta = {False: 1, True:beta}[beta != None]
        self.new_price = np.zeros(self.asset_resolution)
        self.old_price =np.zeros(self.asset_resolution)
        self.history = np.zeros((self.asset_resolution, self.time_resolution))

    @abstractmethod
    def setup_boundary_conditions(self, time):
        ...
    
    @abstractmethod
    def setup_initial_conditions(self):
        ...

    def add_reaction(update_method):
        @wraps(update_method)
        def reaction_wrapper(self):
            update_method(self)
            for s in range(1, len(self.asset_price_range) - 1):
                self.new_price[s] += self.old_price[s]*self.delta_t*self.r
        return reaction_wrapper

    @add_reaction    
    def euler_center_difference(self):
        for s in range(1, len(self.asset_price_range) - 1):
            self.new_price[s] = self.old_price[s] - 0.5*self.courant*self.alpha(s)*(self.old_price[s + 1] -
                self.old_price[s - 1]) + self.fourier*self.beta(s)*(self.old_price[s + 1] - 2*self.old_price[s] - self.old_price[s - 1]) 
    
    def update_price(self, flag='Euler-Center'):
        {'Euler-Center':self.euler_center_difference}[flag]()
    
    def alpha(self, s):
        return self._alpha

    def beta(self, s):
        return self._beta
    
    @property
    def delta_t(self):
        return self.maturity/self.time_resolution
    
    @property
    def delta_s(self):
        return (self.asset_high - self.asset_low)/self.asset_resolution
    
    @property
    def courant(self):
        return self.delta_t/self.delta_s
    
    @property
    def fourier(self):
        return self.delta_t/self.delta_s**2    
    
    def __call__(self, flag='Euler-Center'):
        self.setup_initial_conditions()
        for count, _time in enumerate(np.linspace(0, self.maturity, self.time_resolution)):
            self.setup_boundary_conditions(_time)
            self.update_price(flag)
            self.history[:,count] = self.new_price 
            self.old_price = self.new_price


# ##################################################### #
# #### .  Black-Sholes for Europeam Call Options . #### #
# ##################################################### #
class BlackSholes(ExplicitAdvectionDiffusion):

    def __init__(self, **kwargs):
        super(BlackSholes, self).__init__(**kwargs)
    
    def alpha(self, s):
        return -self.r*self.asset_price_range[s]
    
    def beta(self, s):
        return 0.5*(self.sigma**2)*(self.asset_price_range[s]**2)
        
    def setup_boundary_conditions(self, time):
        self.new_price[0] = 0
        self.new_price[-1] = self.asset_high - self.strike_price*np.exp(-self.r*(self.maturity - time))

    def setup_initial_conditions(self):
        self.old_price = np.array([np.max([s - self.strike_price , 0]) for s in self.asset_price_range])


