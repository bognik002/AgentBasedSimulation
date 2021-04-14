from agents import ExchangeAgent
from tqdm import tqdm
import matplotlib.pyplot as plt


def liquidity_spread(order_book: dict):
    if not order_book['ask'] or not order_book['bid']:
        return None
    return order_book['ask'].first.price - order_book['bid'].first.price


def inventory_quantity(order_book: dict):
    return sum([order.qty for order in order_book['ask']]) - sum([order.qty for order in order_book['bid']])


def calculate_price(spread: dict):
    if spread['bid'] and spread['ask']:
        return (spread['bid'] + spread['ask']) / 2
    else:
        return 0


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
        self.spread = list()  # Essential for Noise and MM Traders
        self.market_volume = list()
        self.liquidity = list()  # bid-ask spread size
        self.mm_inventory = list()
        self.mm_states = list()

    def fit(self, n, nt_lag=0, mm_lag=0):
        self.iterations += n
        iterations = range(n)

        for it in tqdm(iterations, desc='Simulation'):
            # Update variables
            self.spread.append(self.market.spread())
            self.market_volume.append({'bids': len(self.market.order_book['bid']),
                                       'asks': len(self.market.order_book['ask']),
                                       'inventory': inventory_quantity(self.market.order_book)})
            self.liquidity.append(liquidity_spread(self.market.order_book))

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

        return self

    def plot_price(self, show_spread=False, title='Commodity Price', lw=1):
        iterations = range(self.iterations)

        plt.plot(iterations, list(map(calculate_price, self.spread)), color='black', label='mean', lw=lw)
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

    def plot_market_volume(self, lw=1):
        iterations = range(self.iterations)
        plt.title('Market Volume')
        plt.xlabel('Iteration')
        plt.ylabel('Quantity')
        plt.plot(iterations, [val['inventory'] for val in self.market_volume], color='black', lw=lw, label='inventory')
        plt.plot(iterations, [val['bids'] for val in self.market_volume], color='green', lw=lw, label='bids')
        plt.plot(iterations, [val['asks'] for val in self.market_volume], color='red', lw=lw, label='asks')
        plt.legend()
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