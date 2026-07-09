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



    def kappa(self,stockPrices,strikePrices,volatility):
        '''
        Black-Scholes Call Price
        Inputs:
        

        Returns:
        Black-Scholes Price of European Call Option
        
        '''

        l = strikePrices.shape[0]

        taus = self.t-self.times[:-1]
        taus = np.reshape(taus, (1, self.nSteps, 1))
        taus = np.broadcast_to(taus, (self.nSims, self.nSteps, l))

        x = np.reshape(stockPrices, (self.nSims, self.nSteps+1,1))
        x = np.broadcast_to(x, (self.nSims, self.nSteps+1,l))


        d1 = (np.log(x[:,:self.nSteps]/strikePrices) + (self.r - (0.5)*volatility**2)*taus)/(volatility*np.sqrt(taus))

        d2 = d1 + volatility*np.sqrt(taus)

        res = x[:,:self.nSteps]*norm.cdf(d2) - np.exp(-self.r*taus)*strikePrices*norm.cdf(d1)

        res = np.insert(res, self.nSteps, np.maximum(0, x[:,-1] - strikePrices), axis = 1)

        return res


    def delta(self,stockPrices,strikePrices,volatility):
        '''
        Black-Scholes Call Price
        Inputs:
        
        
        Returns:
        Black-Scholes Price of European Call Option
        
        '''
        
        l = strikePrices.shape[0]

        taus = self.t-self.times[:-1]
        taus = np.reshape(taus, (1, self.nSteps, 1))
        taus = np.broadcast_to(taus, (self.nSims, self.nSteps, l))

        x = np.reshape(stockPrices, (self.nSims, self.nSteps+1,1))
        x = np.broadcast_to(x, (self.nSims, self.nSteps+1,l))



        d2 = (np.log(x[:,:self.nSteps]/strikePrices) + (self.r + (0.5)*volatility**2)*taus)/(volatility*np.sqrt(taus))

        res = norm.cdf(d2)

        res = np.insert(res, self.nSteps, 1, axis = 1)

        return res



    def poissonCallPrice(self, x, K, volatility, ltild, error):
        
        if ltild <= 0:
            print('There is arbitrage!')
            raise Exception



        l = K.shape[0]

        taus = self.t-self.times
        taus = np.reshape(taus, (1, self.nSteps+1, 1))
        taus = np.broadcast_to(taus, (self.nSims, self.nSteps+1, l))
        logTaus = np.log(taus[:,:-1])

        x = np.reshape(x, (self.nSims, self.nSteps+1,1))
        x = np.broadcast_to(x, (self.nSims, self.nSteps+1,l))    


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


    def gamma1(self, x, K, volatility, ltild, error):
        temp1 = self.poissonCallPrice((volatility+1)*x, K, volatility, ltild, error)
        temp2 = self.poissonCallPrice(x, K, volatility, ltild, error)

        return (temp1 - temp2)/(volatility * np.reshape(x, (self.nSims, self.nSteps+1,1)))


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

        l = K.shape[0]

        taus = self.t-self.times
        taus = np.reshape(taus, (1, self.nSteps+1, 1))
        taus = np.broadcast_to(taus, (self.nSims, self.nSteps+1, l))
        logTaus = np.log(taus[:,:-1])



        z = x * np.exp(- btild * ltild * (self.t-self.times))
        totalSum = self.kappa(z,K,volatility)



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


        l = K.shape[0]

        taus = self.t-self.times
        taus = np.reshape(taus, (1, self.nSteps+1, 1))
        taus = np.broadcast_to(taus, (self.nSims, self.nSteps+1, l))
        logTaus = np.log(taus[:,:-1])


        z = x * np.exp(- btild * ltild * (self.t-self.times))
        totalSum = self.delta(z,K,volatility)



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

        _, _, l = stockShares.shape
        stockPrice = np.reshape(stockPrice, (self.nSims,self.nSteps+1,1))
        stockPrice = np.broadcast_to(stockPrice, (self.nSims,self.nSteps+1,l))

        ts = np.reshape(self.times[1:], (1, self.nSteps, l))

        res = np.broadcast_to(np.exp(-self.r * ts), (self.nSims,self.nSteps, l))


        res = res * (stockShares[:,0:self.nSteps] - stockShares[:,1:]) * stockPrice[:,1:]
        res = np.cumsum(res, axis = 1)
        res = np.insert(res, 0, 0, axis=1)


        return res




    def calculateDiscountedPortfolioValues(self,
                                           initialCapital,
                                           stockPrice,
                                           stockShares):

        _, _, l = stockShares.shape
        stockPrice = np.reshape(stockPrice, (self.nSims,self.nSteps+1,1))
        stockPrice = np.broadcast_to(stockPrice, (self.nSims,self.nSteps+1,l))


        profits = self.calculateDiscountedStockProfits(stockPrice,stockShares)

        profits = profits + np.reshape((initialCapital - stockShares[0][0] * stockPrice[0][0]), (self.nSims,1,l))


        discountedStockValues = self.discountPrices(stockPrice * stockShares)


        return profits + discountedStockValues



    def discountPrices(self, prices):
        _, _, l = prices.shape
        discounts = np.reshape(np.exp(-self.r * self.times), (1,self.nSteps+1, 1))
        return prices * np.broadcast_to(discounts, (self.nSims, self.nSteps+1, l))



    def plotSimulation(self, process, ax):
        for i in range(self.nSims):
            ax.plot(self.times,process[i])
