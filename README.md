# Jump-Models
(Quantitative Finance Bootcamp Summer 2026, The Erdos Institute)

## Team Members 
Manav Batavia, Henry Fontana, Mohammad Tariquel Islam, Shin Kim, Xuzhi Tang

## Repository Structure 
 - `Jump.ipynb` notebook containing simulations of stock paths, call option paths, and hedging strategies. 
 - `ParameterOptimizationOnSimulatedData.ipynb` notebook containing parameter calibration for different models.
 - `StockPathSimulation.py` python code for simulating stock paths.
 - `StrategySimulation.py` python code for computing call option prices and implementing hedging strategies. 

## Overview
In this project we study jump diffusion models for asset prices. Whether or not stock paths exhibit jumps has implications for risk management and asset allocation. For instance, call options are priced differently if the underlying stock path exhibits jumps. Moreover, empirical work suggests that jumps are common in financial markets. We study jump diffusion models with constant volatility and mean rate of return. We simulate stock paths and call option prices under different models. Additionally, we simulate hedging strategies for the call option under the different models, and calculate the expected portfolio value for a hedging strategy under the jump diffusion model via the Monte Carlo method. Finally, we calibrate the parameters of the jump diffusion model on simulated minute-by-minute data.

## Methodology and Results
### Simulation of stock paths, call option paths, and hedging strategies
We use the StockPathSimulation class and StrategySimulation class from StockPathSimulation.py and StrategySimulation.py to simulate stock prices, call option prices, and hedging portfolios. Monte Carlo simulations are used throughout to generate sample paths and evaluate pricing and hedging performance. The formulas for simulating stock prices and computing option prices are found in ``Stochastic Calculus for Finance 2" by Steven E. Shreve. The call option prices for models with jumps involve an infinite sum. However, we can truncate the series to a sum with a relatively small number of summands (~20 summands) while incurring only a small error ($\epsilon = 1\times 10^{-6}$). Chat GPT was consulted for this approximation..

### Parameter calibration
Model parameters, including diffusion volatility, jump intensities, and jump sizes, are estimated using nonlinear least-squares optimization. Parameter calibration was conducted in two different ways. First, we made random guesses for the initial input parameters in least squares regression. While this is effective in simpler models with low dimensional parameter space, least squares regression often fails to converge when calibrating parameters for the jump diffusion model. To address this issue, we use the ideas from ``Econometrics of Testing for Jumps in Financial Economics Using Bipower Variation" by Ole E. Barndorff-Nielsen and Neil Shephard and the k-means clustering algorithm to create ``informed" guesses of the parameters to use in least squares regression. Chat GPT was consulted in implementing this strategy.

## Results
Making ``informed" for the initial parameters improved the convergence rate of least squares from ~25% to ~55% for jump diffusion models with three different jump sizes. Additionally, there was a ~40% decrease in the time it took for least squares to converge.

## Future Directions 
So far, we have only considered simulated minute-by-minute data. It would be insightful to see how our parameter calibration method performs on real world data. Additionally, our model assumes constant volatility. The work in ``Econometrics of Testing for Jumps in Financial Economics Using Bipower Variation" also applies to models with volatility that varies over time. It would be interesting to generalize our implementation to account for situations with varying volatility.


## References

 - Ole E. Barndorff-Nielsen and Neil Shephard. Econometrics of Testing for Jumps in Financial Economics Using Bipower Variation. Journal of Financial Econometrics.
 - Steven E. Shreve. Stochastic Calculus for Finance II. Springer Finance Textbook.
