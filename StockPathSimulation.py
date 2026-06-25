import numpy as np


class StockPathSimulation:

    def __init__(self,
                 expiryTime = 1,
                 numOfSims = 10,
                 numOfSteps = 252):
        self.t = expiryTime
        self.nSims = numOfSims
        self.nSteps = numOfSteps
    


    def simBrownianMotionProcess(self):

        increment = self.t/self.nSteps

        res = np.random.normal(loc = 0, scale = np.sqrt(increment), size = (self.nSims,self.nSteps))
        res = np.insert(res, 0, 0, axis=1)
        return np.cumsum(res, axis = 1)
    
    
    def simPoissonProcess(self, intensity = 1.0):

        increment = self.t/self.nSteps

        res = np.random.poisson(lam = intensity * increment, size = (self.nSims,self.nSteps))
        res = np.insert(res, 0, 0, axis=1)
        return np.cumsum(res, axis = 1)


    def simCompoundPoissonProcess(self,
                                  intensity = 1.0,
                                  jumps = [-1, 1],
                                  jumpProbabilities = [0.5, 0.5]):
        res = np.zeros((self.nSims, self.nSteps+1))
        for i in range(len(jumps)):
            res += jumps[i] * self.simPoissonProcess(intensity = jumpProbabilities[i] * intensity)

        return res


    def assetModel1(self,
                    meanRateOfReturn = 1.0,
                    volatility = 0.5,
                    intensity = 1.0,
                    initialPrice = 100.0):
        ts = np.linspace(0,self.t,self.nSteps+1)
        res, _ = np.meshgrid(ts, np.zeros(self.nSims))

        res = res * (meanRateOfReturn - volatility*intensity)
        res += np.log(volatility + 1) * self.simPoissonProcess(intensity)

        return initialPrice * np.exp(res)




    def assetModel2(self,
                   meanRateOfReturn = 1.0,
                   volatility = 0.5,
                   intensity = 1.0,
                   jumps = [-0.01, 0.01],
                   jumpProbabilities = [0.5, 0.5],
                   initialPrice = 100.0):
        beta = np.dot(jumps, jumpProbabilities)
        res = volatility * self.simBrownianMotionProcess()
        res += np.array(np.linspace(0,self.t,self.nSteps+1)) * (meanRateOfReturn - beta*intensity - 0.5*volatility**2)
        res += self.simCompoundPoissonProcess(intensity, np.log(np.array(jumps)+1), jumpProbabilities)
        return initialPrice * np.exp(res)


