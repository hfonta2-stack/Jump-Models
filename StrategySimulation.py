import numpy as np

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




