from agents import ExchangeAgent
from tqdm import tqdm
import matplotlib.pyplot as plt


def liquidity_spread(order_book):
    if not order_book['ask'] or not order_book['bid']:
        return None
    return order_book['ask'].first.price - order_book['bid'].first.price


class Simulator:
    def __init__(self, market: ExchangeAgent, noise_agents=None, market_makers=None):
        """
        :param market: ExchangeAgent
        :param noise_agents: NoiseAgent
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
        self.mm_inventory = list()
        self.mm_states = list()

    def fit(self, n, nt_lag=0, mm_lag=0):
        self.iterations += n
        iterations = range(n)

        self.spread.append(self.market.spread())  # Spread 0 iteration
        for it in tqdm(iterations, desc='Simulation'):
            # Call MarketMakers
            if self.market_makers:
                for trader in self.market_makers:
                    self.mm_inventory.append({'qty': trader.inventory, 'name': trader.name})
                    self.mm_states.append({'state': trader.state, 'name': trader.name})
                    trader.call(self.spread[max(it - mm_lag, 0)])

            # Call NoiseAgents
            if self.noise_traders:
                for trader in self.noise_traders:
                    trader.call(self.spread[max(it - nt_lag, 0)])

            # Update variables
            self.spread.append(self.market.spread())
            self.market_volume.append(sum([order.qty for order in self.market.order_book['bid']]) +
                                      sum([order.qty for order in self.market.order_book['ask']]))
            self.liquidity.append(liquidity_spread(self.market.order_book))

        return self

    def plot_price(self, show_spread=False, title='Commodity Price', lw=1):
        iterations = range(self.iterations + 1)
        spread_history = list()
        for spread in self.spread:
            if spread['bid'] and spread['ask']:
                spread_history.append((spread['bid'] + spread['ask']) / 2)
            elif spread['bid']:
                spread_history.append(spread['bid'])
            elif spread['ask']:
                spread_history.append(spread['ask'])
            else:
                spread_history.append(0)

        plt.plot(iterations, spread_history,
                 color='black', label='mean', lw=lw)
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

    def plot_inventory(self, lw=1):
        iterations = range(self.iterations)
        plt.title('Agents Inventories')
        plt.xlabel('Iteration')
        plt.ylabel('Inventory')
        for trader in self.market_makers:
            plt.plot(iterations, [val['qty'] for val in self.mm_inventory if trader.name == val['name']],
                     lw=lw, label=trader.name)
        plt.legend()
        plt.show()