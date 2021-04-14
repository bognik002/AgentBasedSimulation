from agents import ExchangeAgent, NoiseAgent, MarketMaker
from simulator import Simulator

spread_initial = {'bid': 200, 'ask': 250}

# Initialize Exchange Agent
exchange = ExchangeAgent(spread_initial, depth=1000)

# Initialize Traders
noise_agents = [NoiseAgent(exchange) for i in range(10)]
# market_makers = [MarketMaker(exchange, 0, 10, -10)]

# Simulation
simulator = Simulator(exchange, noise_agents=noise_agents, market_makers=None).fit(100)
simulator.plot_price()
# simulator.plot_liquidity()
simulator.plot_market_volume()
# simulator.plot_inventory()
