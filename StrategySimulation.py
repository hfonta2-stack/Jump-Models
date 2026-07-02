import numpy as np
from scipy.stats import poisson, norm

class StrategySimulation:

    def __init__(self,
                 expirationTime = 1.0,
                 numOfSims = 10,
                 numOfSteps = 252,
                 interestRate = 0.05):
        self.t = expirationTime
        self.nSims = numOfSims
        self.nSteps = numOfSteps
        self.r = interestRate
        self.times = np.linspace(0,expirationTime,numOfSteps+1)



    def kappa(self,x,K,volatility):
        '''
        Black-Scholes Call Price
        Inputs:
        

        Returns:
        Black-Scholes Price of European Call Option
        
        '''


        taus = np.broadcast_to(self.t-self.times[:-1], (self.nSims,self.nSteps))


        d1 = (np.log(x[:,:self.nSteps]/K) + (self.r - (0.5)*volatility**2)*taus)/(volatility*np.sqrt(taus))

        d2 = d1 + volatility*np.sqrt(taus)

        res = x[:,:self.nSteps]*norm.cdf(d2) - np.exp(-self.r*taus)*K*norm.cdf(d1)

        res = np.insert(res, self.nSteps, np.maximum(0, x[:,-1] - K), axis = 1)

        return res


    def delta(self,x,K,volatility):
        '''
        Black-Scholes Call Price
        Inputs:
        
        
        Returns:
        Black-Scholes Price of European Call Option
        
        '''

        taus = np.broadcast_to(self.t-self.times[:-1], (self.nSims,self.nSteps))

        d2 = (np.log(x[:,:self.nSteps]/K) + (self.r + (0.5)*volatility**2)*taus)/(volatility*np.sqrt(taus))

        res = norm.cdf(d2)

        res = np.insert(res, self.nSteps, 1, axis = 1)

        return res



    def calculateDiscountedStockProfits(self,
                                   stockPrice,
                                   stockShares):

        times = np.linspace(0,self.t,self.nSteps+1)
        res = np.broadcast_to(np.exp(-self.r * times[1:]), (self.nSims,self.nSteps))


        res = res * (stockShares[:,0:self.nSteps] - stockShares[:,1:]) * stockPrice[:,1:]
        res = np.cumsum(res, axis = 1)
        res = np.insert(res, 0, 0, axis=1)


        return res




    def calculateDiscountedPortfolioValues(self,
                                           initialCapital,
                                           stockPrice,
                                           stockShares):


        times = np.linspace(0,self.t,self.nSteps+1)

        profits = self.calculateDiscountedStockProfits(stockPrice,stockShares)

        profits = profits + np.reshape((initialCapital - stockShares[0][0] * stockPrice[0][0]), (self.nSims,1))


        discountedStockValues = np.broadcast_to(np.exp(-self.r * times), (self.nSims,self.nSteps+1))* stockPrice * stockShares


        return profits + discountedStockValues





    def plotSimulation(self, process, ax):
        for i in range(self.nSims):
            ax.plot(self.times,process[i])
