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
                      probe_trader=None).fit(500, nt_lag=0, mm_lag=0)

market_states = simulator.info.market_states()
price_states = simulator.info.price_states(.7)
volume_states = simulator.info.volume_states(.7)
volatility_states = simulator.info.volatility_states(.3)
print('market, price, volume, volatility')
for i in range(len(market_states)):
    print(i, market_states[i], price_states[i], volume_states[i], volatility_states[i], sep='\t\t')

simulator.plot_price()
simulator.plot_volume()
simulator.plot_volatility()
simulator.plot_inventory(lw=.5)
simulator.plot_states_heatmap()
