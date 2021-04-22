from agents import ExchangeAgent, NoiseAgent, MarketMaker
from simulator import Simulator

spread_initial = {'bid': 100, 'ask': 200}

# Initialize Exchange Agent
exchange = ExchangeAgent(spread_initial, depth=1000)

# Initialize Traders
noise_agents = [NoiseAgent(exchange) for i in range(100)]
market_makers = [MarketMaker(exchange, 0, 100, -100) for j in range(2)]


# Simulation
simulator = Simulator(exchange, noise_agents=noise_agents, market_makers=market_makers).fit(1000, nt_lag=2, mm_lag=2)
simulator.plot_price()
simulator.plot_market_volume()
simulator.plot_liquidity()
simulator.plot_inventory()

