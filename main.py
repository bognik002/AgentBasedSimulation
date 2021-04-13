from agents import ExchangeAgent, NoiseAgent, MarketMaker
from simulator import Simulator
import matplotlib.pyplot as plt

spread_initial = {'bid': 200, 'ask': 250}
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
exchange = ExchangeAgent(spread_init=spread_initial, depth=1000, **market_params)

# Initialize Traders
noise_traders = [NoiseAgent(exchange, i, 200, **noise_params) for i in range(100)]
market_makers = [MarketMaker(exchange, i, 1000, spread_initial['ask'], spread_initial['bid']) for i in range(1)]

# Simulation
for mm_l in range(0, 50, 10):
    simulator = Simulator(exchange, noise_agents=noise_traders, market_makers=market_makers).fit(500, nt_lag=0, mm_lag=mm_l)
    simulator.plot_price(show_spread=False)
    plt.savefig(f'pictures/price(mm_lag={mm_l}).png')
    plt.show()
    simulator.plot_market_volume()
    plt.savefig(f'pictures/volume(mm_lag={mm_l}).png')
    plt.show()
    simulator.plot_inventory()
    plt.savefig(f'pictures/inventory(mm_lag={mm_l}).png')
    plt.show()

