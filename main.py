from agents import ExchangeAgent, NoiseAgent, MarketMaker
from simulator import Simulator

spread_initial = {'bid': 200, 'ask': 250}
market_params = {'price_std': 1, 'quantity_mean': 0, 'quantity_std': 2}
noise_params = {'price_std': 1, 'quantity_mean': 0, 'quantity_std': 2}

# Initialize Exchange Agent
exchange = ExchangeAgent(spread_initial, depth=10000)

# Initialize Traders
noise_traders = [NoiseAgent(exchange) for i in range(1)]

# Simulation
simulator = Simulator(exchange, noise_agents=noise_traders).fit(10000, nt_lag=0)
