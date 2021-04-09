from agents import ExchangeAgent, NoiseAgent, MarketMaker
from simulator import Simulator
import matplotlib.pyplot as plt

spread_initial = {'bid': 200, 'ask': 201}
market_params = {
    'lambda_': .2,
    'mu': 0,
    'sigma': .2
}

noise_params = {
    'lambda_': 1,
    'mu': 0,
    'sigma': 1
}

# Initialize Exchange Agent
exchange = ExchangeAgent(spread_init=spread_initial, depth=20000, **market_params)

# Initialize Traders
noise_traders = [NoiseAgent(exchange, i, 200, **noise_params) for i in range(1000)]
market_makers = [MarketMaker(exchange, i, 1000, 200, 2000) for i in range(10)]

# Simulation
simulator = Simulator(exchange, noise_agents=noise_traders, market_makers=market_makers).fit(1200, nt_lag=1, mm_lag=1)
simulator.plot_price(show_spread=True)
simulator.plot_market_volume()
simulator.plot_inventory(market_makers)
plt.show()
