from agents import ExchangeAgent, NoiseAgent, MarketMaker
from simulator import Simulator
import matplotlib.pyplot as plt

spread_initial = {'bid': 200, 'ask': 250}
market_params = {'price_std': 1, 'quantity_mean': 0, 'quantity_std': 2}
noise_params = {'price_std': 1, 'quantity_mean': 0, 'quantity_std': 2}

# Initialize Exchange Agent
exchange = ExchangeAgent(spread_initial, **market_params)

# Initialize Traders
noise_traders = [NoiseAgent(exchange, **noise_params) for i in range(1)]
#market_makers = [MarketMaker(exchange, 1000, spread_initial['ask'], spread_initial['bid']) for j in range(2)]

# Simulation
simulator = Simulator(exchange, noise_agents=noise_traders).fit(1000, nt_lag=0, mm_lag=0)
simulator.plot_price()
