from agents import ExchangeAgent, NoiseAgent, MarketMaker, ProbeTrader
from simulator import Simulator


spread_initial = {'bid': 500, 'ask': 600}

# Initialize Exchange Agent
exchange = ExchangeAgent(spread_initial, depth=1000)

# Initialize Traders
noise_agents = [NoiseAgent(exchange) for i in range(100)]
market_makers = [MarketMaker(exchange, 45, 100, 10) for j in range(2)]
probe_trader = ProbeTrader(exchange, 'ask', quantity_mean=6)

# Simulation
simulator = Simulator(exchange, noise_agents=noise_agents, market_makers=market_makers,
                      probe_trader=None).fit(1000)
simulator.plot_states_heatmap()
