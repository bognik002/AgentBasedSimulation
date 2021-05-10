from agents import ExchangeAgent, NoiseAgent, MarketMaker, ProbeTrader
from simulator import Simulator

spread_initial = {'bid': 100, 'ask': 200}

# Initialize Exchange Agent
exchange = ExchangeAgent(spread_initial, depth=1000)

# Initialize Traders
noise_agents = [NoiseAgent(exchange) for i in range(100)]
market_makers = [MarketMaker(exchange, 0, 100, -100) for j in range(2)]
probe_trader = ProbeTrader(exchange, 'ask', quantity_mean=2, quantity_std=1)


# Simulation
simulator = Simulator(exchange, noise_agents=noise_agents, market_makers=market_makers,
                      probe_trader=probe_trader).fit(1000, nt_lag=0, mm_lag=0)
simulator.plot_market_volume()
simulator.plot_price()
simulator.plot_states()
simulator.plot_inventory()
