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

        taus = self.t-self.times[:-1]
        taus = np.broadcast_to(taus, (self.nSims, self.nSteps))


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

        taus = self.t-self.times[:-1]
        taus = np.broadcast_to(taus, (self.nSims, self.nSteps))

        d2 = (np.log(x[:,:self.nSteps]/K) + (self.r + (0.5)*volatility**2)*taus)/(volatility*np.sqrt(taus))

        res = norm.cdf(d2)

        res = np.insert(res, self.nSteps, 1, axis = 1)

        return res



    def poissonCallPrice(self, x, K, volatility, meanRate, intensity, error):

        ltild = intensity - (meanRate - self.r) / volatility
        if ltild <= 0:
            print('There is arbitrage!')
            raise Exception
    

        taus = np.broadcast_to(self.t - self.times, (self.nSims,self.nSteps+1))
        logTaus = np.log(taus[:,:-1])

        logFactorial = 0

        stockTerm = x * np.exp(- ltild * volatility * taus)
        strikeTerm = K * np.exp(-self.r * taus)

        total_sum = np.maximum(0.0, stockTerm - strikeTerm)

        priceCap = x[0,0]*10
        jMax = np.ceil(np.max(poisson.ppf(1 - error/priceCap, ltild * (1 + volatility) * taus)))


        j = 1
        while j <= jMax:

            payoffTerm = np.maximum(0.0, stockTerm * np.exp(j * np.log(volatility + 1)) - strikeTerm)

            logProb = j* np.log(ltild) + j* logTaus - logFactorial
            weight = np.insert(np.exp(logProb), self.nSteps, 0, axis = 1) 

            total_sum += payoffTerm * weight

            j += 1
            logFactorial += np.log(j)

        return total_sum * np.exp(- ltild * taus)


    def gamma1(self, x, K, volatility, meanRate, intensity, error):

        temp1 = self.poissonCallPrice((volatility+1)*x, K, volatility, meanRate, intensity, error)
        temp2 = self.poissonCallPrice(x, K, volatility, meanRate, intensity, error)
        return (temp1 - temp2)/(volatility * x)


    def findCombinations(self, j, numJumps):
        res = []

        def dfs(cur, curInd, curTotal):
            if curTotal == j:
                res.append(cur.copy())
                return
            if curInd >= numJumps:
                return
            cur[curInd] += 1
            dfs(cur,curInd,curTotal + 1)
            cur[curInd] -= 1

            dfs(cur,curInd + 1, curTotal)

            return
        
        cur = [0 for _ in range(numJumps)]

        dfs(cur,0,0)

        return res

                





    def jumpDiffusionCallPrice(self, x, K, volatility, jumps, parameters, error):

        ltild = np.sum(parameters)
        p = parameters/ltild
        btild = p @ jumps

        taus = np.broadcast_to(self.t - self.times, (self.nSims,self.nSteps+1))
        z = x * np.exp(- btild * ltild * taus)
        totalSum = self.kappa(z,K,volatility)


        logTaus = np.log(taus[:,:-1])


        priceCap = x[0,0]*10
        jMax = int(np.ceil(np.max(poisson.ppf(1 - error/priceCap, ltild * (1 + btild) * taus))))

        logFactorials = [0 for _ in range(jMax + 1)]
        
        for i in range(jMax):
            logFactorials[i+1] = logFactorials[i] + np.log(i+1)


        a = len(jumps)
        y = np.array(jumps)


        j = 1


        while j <= jMax:

            combos = self.findCombinations(j,a)
            
            payoffTerm = 0

            for combo in combos:
                temp = logFactorials[j]
                for k in combo:
                    temp -= logFactorials[k]
                temp = np.prod(np.power(p,combo)) * np.exp(temp)
                payoffTerm +=  self.kappa(z * np.prod(np.power(1+y, combo)),K,volatility) * temp

            logProb = j* np.log(ltild) + j* logTaus - logFactorials[j]
            weight = np.insert(np.exp(logProb), self.nSteps, 0, axis = 1) 

            totalSum += payoffTerm * weight


            j += 1


        return totalSum * np.exp(- ltild * taus)




    def gamma2(self, x, K, volatility, jumps, parameters, error):

        ltild = np.sum(parameters)
        p = parameters/ltild
        btild = p @ jumps

        taus = np.broadcast_to(self.t - self.times, (self.nSims,self.nSteps+1))
        z = x * np.exp(- btild * ltild * taus)
        totalSum = self.delta(z,K,volatility)


        logTaus = np.log(taus[:,:-1])

        priceCap = x[0,0]*10
        jMax = int(np.ceil(np.max(poisson.ppf(1 - error/priceCap, ltild * (1 + btild) * taus))))


        logFactorials = [0 for _ in range(jMax + 1)]
        
        for i in range(jMax):
            logFactorials[i+1] = logFactorials[i] + np.log(i+1)


    

        a = len(jumps)
        y = np.array(jumps)


        j = 1


        while j <= jMax:

            combos = self.findCombinations(j,a)
            
            payoffTerm = 0

            for combo in combos:
                temp = logFactorials[j]
                for k in combo:
                    temp -= logFactorials[k]
                temp = np.prod(np.power(p,combo)) * np.exp(temp) * np.prod(np.power(1+y, combo))
                payoffTerm +=  self.delta(z * np.prod(np.power(1+y, combo)),K,volatility) * temp
 



            logProb = j* np.log(ltild) + j* logTaus - logFactorials[j]
            weight = np.insert(np.exp(logProb), self.nSteps, 0, axis = 1) 

            totalSum += payoffTerm * weight

            j += 1
            
        return totalSum * np.exp(- ltild *(1 + btild) * taus)






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
