#Reference:(Python for Finance, 2nd ed.,Dr. Yves J. Hilpisch)
import numpy as np
from scipy import stats


class bsm_option(object):
    ''' Class for European call and put options in BSM model.

    Attributes
    ==========
    option: str
        option type (call, put)    
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        maturity (in year fractions)
    r: float
        constant risk-free short rate
    sigma: float
        volatility factor in diffusion term

    Methods
    =======
    value: float
        returns the present value of call option
    vega: float
        returns the Vega of call option
    imp_vol: float
        returns the implied volatility given option quote
    '''

    def __init__(self, S0, K, r,T, sigma,option='Put'):
        self.option=option
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.d1 = ((np.log(self.S0 / self.K) +
               (self.r + 0.5 * self.sigma ** 2) * self.T) /
              (self.sigma * np.sqrt(self.T)))
        self.d2 = ((np.log(self.S0 / self.K) +
               (self.r - 0.5 * self.sigma ** 2) * self.T) /
              (self.sigma * np.sqrt(self.T)))

    def value(self):
        ''' Returns option value.
            Set ds=True to return d1 and d2
        ''' 
        call = (self.S0 * stats.norm.cdf(self.d1, 0.0, 1.0) -
                 self.K * np.exp(-self.r * self.T) * stats.norm.cdf(self.d2, 0.0, 1.0))
        put=call+self.K*np.exp(-self.r*self.T)-self.S0
        
        if self.option=='Call':
            return call
        elif self.option=='Put': 
            return put
        return
    
    def Delta(self):
        ''' Returns Delta of given option price.
        '''
        if self.option=='Call':
            return stats.norm.cdf(self.d1, 0.0, 1.0)
        elif self.option=='Put': 
            return (stats.norm.cdf(self.d1, 0.0, 1.0)-1)
        
    def Gamma(self):
        ''' Returns Gamma of given option price.
        '''
        return stats.norm.pdf(self.d1, 0.0, 1.0)/(self.S0*self.sigma*np.sqrt(self.T))
       
    def Theta(self):
        ''' Returns Theta of given option price.
        '''
        if self.option=='Call':
            return -(stats.norm.pdf(self.d1, 0.0, 1.0)*self.S0*self.sigma)/(2*np.sqrt(self.T))-(self.r*self.K*np.exp(-self.r*self.T)*stats.norm.cdf(self.d2, 0.0, 1.0))
        elif self.option=='Put': 
            return (-(stats.norm.pdf(self.d1, 0.0, 1.0)*self.S0*self.sigma)/(2*np.sqrt(self.T)))+(self.r*self.K*np.exp(-self.r*self.T)*(1-stats.norm.cdf(self.d2, 0.0, 1.0)))
    def Vega(self):
        ''' Returns Vega of option.
        '''
        return self.S0 * stats.norm.pdf(self.d1, 0.0, 1.0) * np.sqrt(self.T)

    def imp_vol(self, Pr0, sigma_est=0.2, it=100):
        ''' Returns implied volatility given option price.
        '''
        option = bsm_option(self.option,self.S0, self.K, self.T, self.r, sigma_est)
        for i in range(it):
            option.sigma -= (option.value() - Pr0) / option.vega()         
        return option.sigma
    
    def Rho(self):
        ''' Returns Rho of given option price.
        '''
        if self.option=='Call':
            return self.K*self.T*np.exp(-self.r*self.T)*stats.norm.cdf(self.d2, 0.0, 1.0)
        elif self.option=='Put':
            return -self.K*self.T*np.exp(-self.r*self.T)*(1-stats.norm.cdf(self.d2, 0.0, 1.0))
    
