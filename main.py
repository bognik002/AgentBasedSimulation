from agents import ExchangeAgent, NoiseAgent, MarketMaker
from simulator import Simulator

spread_initial = {'bid': 200, 'ask': 250}
market_params = {'price_std': 1, 'quantity_mean': 0, 'quantity_std': 2}
noise_params = {'price_std': 1, 'quantity_mean': 0, 'quantity_std': 2}

# Initialize Exchange Agent
exchange = ExchangeAgent(spread_initial, depth=1000)

# Initialize Traders
noise_traders = [NoiseAgent(exchange) for i in range(10)]
market_makers = [MarketMaker(exchange, 20, 25, 15) for j in range(2)]

# Simulation
simulator = Simulator(exchange, noise_agents=noise_traders, market_makers=market_makers).fit(100, nt_lag=1, mm_lag=1)
simulator.plot_price()
simulator.plot_liquidity()
simulator.plot_market_volume()
simulator.plot_inventory()
