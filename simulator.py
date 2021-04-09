from agents import ExchangeAgent
from tqdm import tqdm
import matplotlib.pyplot as plt


def liquidity_spread(order_book):
    return order_book['ask'][0]['price'] - order_book['bid'][0]['price']


class Simulator:
    def __init__(self, market: ExchangeAgent, noise_agents=None, market_makers=None):
        """
        :param market: ExchangeAgent
        :param noise_agents: NoiseAgent

        spread: list() - [{'bid': ###, 'ask': ###}, ...]
        memory_usage: len of order book for both sides
        """

        # Agents
        self.market = market
        self.noise_traders = noise_agents
        self.market_makers = market_makers

        # Data to store
        self.iterations = 0
        self.spread = list()
        self.memory_usage = list()
        self.market_volume = list()
        self.liquidity = list()

    def fit(self, n, nt_lag=0, mm_lag=0):
        self.iterations += n
        iterations = range(n)

        self.spread.append(self.market.spread())  # Spread 0 iteration
        for it in tqdm(iterations, desc='Simulation'):
            # Call NoiseAgents
            if self.noise_traders:
                for trader in self.noise_traders:
                    trader.call(self.spread[max(it - nt_lag, 0)])
            # Call MarketMakers
            if self.market_makers:
                for trader in self.market_makers:
                    trader.call(self.spread[max(it - mm_lag, 0)])

            # Update variables
            self.spread.append(self.market.spread())
            self.memory_usage.append(len(self.market.order_book['bid']) + len(self.market.order_book['ask']))
            self.market_volume.append(sum([order['qty'] for order in self.market.order_book['bid']]) +
                                      sum([order['qty'] for order in self.market.order_book['ask']]))
            self.liquidity.append(liquidity_spread(self.market.order_book))

        return self

    def plot_price(self, show_spread=False, title='Commodity Price', lw=1):
        iterations = range(self.iterations + 1)
        plt.plot(iterations, [(spread['bid'] + spread['ask']) / 2 for spread in self.spread],
                 color='black', label='mean', ls='--', lw=lw)
        if show_spread:
            plt.plot(iterations, [spread['bid'] for spread in self.spread], color='green', label='bid', lw=lw)
            plt.plot(iterations, [spread['ask'] for spread in self.spread], color='red', label='ask', lw=lw)

        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_liquidity(self, lw=1):
        iterations = range(self.iterations)
        plt.title('Liquidity')
        plt.xlabel('Iteration')
        plt.ylabel('Liquidity')
        plt.plot(iterations, self.liquidity, color='black', lw=lw)
        plt.show()

    def plot_order_book(self):
        self.market.plot_order_book()
        plt.show()

    def plot_memory_usage(self, lw=1):
        iterations = range(self.iterations)
        plt.title('Memory Usage')
        plt.xlabel('Iteration')
        plt.ylabel('N')
        plt.plot(iterations, self.memory_usage, color='black', lw=lw)
        plt.show()

    def plot_market_volume(self, lw=1):
        iterations = range(self.iterations)
        plt.title('Market Volume')
        plt.xlabel('Iteration')
        plt.ylabel('Quantity')
        plt.plot(iterations, self.market_volume, color='black', lw=lw)
        plt.show()

    def plot_inventory(self, agents, lw=1):
        iterations = range(self.iterations + 1)
        plt.title('Agents Inventories')
        plt.xlabel('Iteration')
        plt.ylabel('Inventory')
        for trader in agents:
            plt.plot(iterations, trader.inventory_states, lw=lw, label=trader.name)
        plt.legend()
        plt.show()