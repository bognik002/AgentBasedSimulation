from agents import ExchangeAgent, MarketMaker
from tqdm import tqdm
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.nonparametric.smoothers_lowess import lowess


class Simulator:
    def __init__(self, market: ExchangeAgent, noise_agents=None, market_makers=None, probe_trader=None):
        """
        :param market: ExchangeAgent
        :param noise_agents: NoiseAgent
        """

        # Agents
        self.market = market
        self.noise_traders = noise_agents
        self.market_makers = market_makers
        self.probe_trader = probe_trader

        # Info
        self.info = SimulatorInfo(self)

    def fit(self, n, nt_lag=0, mm_lag=0, probe_start=0, probe_stop=None):
        for it in tqdm(range(n), desc='Simulation'):
            # Capture market state
            self.info.capture()

            # Call MarketMakers
            if self.market_makers:
                for trader in self.market_makers:
                    spread = self.info.spread[max(it - mm_lag, 0)]
                    inventory = self.info.inventories[max(it - mm_lag, 0)][trader.name]
                    trader.call(spread, inventory)

            # Call NoiseAgents
            if self.noise_traders:
                for trader in self.noise_traders:
                    spread = self.info.spread[max(it - nt_lag, 0)]
                    trader.call(spread)

            # Call ProbeTrader
            if self.probe_trader:
                if probe_stop:
                    if probe_start <= it <= probe_stop:
                        self.probe_trader.call()
                else:
                    if it > probe_start:
                        self.probe_trader.call()

        return self

    # Visuals
    def plot_price(self, show_spread=False, smoothing=False, title='Commodity Price', lw=1):
        if not smoothing:
            plt.plot(self.info.iterations, self.info.center_price(), color='black', label='mean', lw=lw)
            if show_spread:
                plt.plot(self.info.iterations, self.info.best_price('bid'), color='green', label='bid', lw=lw)
                plt.plot(self.info.iterations, self.info.best_price('ask'), color='red', label='ask', lw=lw)
        else:
            plt.plot(self.info.iterations, lowess(self.info.center_price(), self.info.iterations, return_sorted=False),
                     color='black', label='mean', lw=lw)
            plt.ylim(min(self.info.center_price()), max(self.info.center_price()))

        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_volume(self, lw=1, smoothing=False):
        plt.title('Market Volume')
        plt.xlabel('Iteration')
        plt.ylabel('Quantity')
        if not smoothing:
            plt.plot(self.info.iterations, self.info.excess_volume(), color='black', lw=lw, label='inventory')
        else:
            plt.plot(self.info.iterations, lowess(self.info.excess_volume(), self.info.iterations, return_sorted=False),
                     color='black', lw=lw, label='inventory (loess)')

        plt.plot(self.info.iterations, self.info.sum_quantity('bid'), color='green', lw=lw, label='bids')
        plt.plot(self.info.iterations, self.info.sum_quantity('ask'), color='red', lw=lw, label='asks')
        plt.legend()
        plt.show()

    def plot_inventory(self, lw=1, smoothing=False):
        plt.title('Agents Inventories')
        plt.xlabel('Iteration')
        plt.ylabel('Inventory')
        for trader in self.market_makers:
            if not smoothing:
                plt.plot(self.info.iterations, self.info.trader_inventory(trader),
                         lw=lw, label=trader.name)
            else:
                plt.plot(self.info.iterations, lowess(self.info.trader_inventory(trader), self.info.iterations,
                                                      return_sorted=False), lw=lw, label=trader.name)
        plt.legend()
        plt.show()

    def plot_panic(self, lw=1):
        plt.plot(self.info.iterations, self.info.panic_count(), lw=lw, color='black')
        plt.title('Market Makers States')
        plt.ylabel('Panic')
        plt.show()

    def plot_volatility(self, lw=1):
        plt.title('Volatility')
        plt.xlabel('Iteration')
        plt.ylabel('std')
        plt.plot(self.info.iterations, self.info.volatility(), lw=lw)
        plt.show()

    def plot_states_heatmap(self):
        order = ['i i sta', 'i d sta', 'd i sta', 'd d sta',
                 'i i vol', 'i d vol', 'd i vol', 'd d vol']
        states = self.info.market_states()
        table = SimulatorInfo.states_markov(states, order=order)
        sns.heatmap(table, annot=True, fmt='g', cmap="Blues", cbar=False)
        plt.show()

    # Tables
    def market_states(self):
        recordings = {
            'price': self.info.center_price().round(2)[:-1],
            'volume': self.info.excess_volume().round(2)[:-1],
            'B_price': self.info.price_states(),
            #'B_volume': self.info.volume_states(.3),
            #'B_vol': self.info.volatility_states(.3)
        }
        table = pd.DataFrame(recordings)
        table.index.name = 'iter'
        return table


class SimulatorInfo:
    def __init__(self, simulator: Simulator):
        # Agents
        self.market = simulator.market
        self.noise_agents = simulator.noise_traders
        self.market_makers = simulator.market_makers
        self.probe_trader = simulator.probe_trader

        # Important
        self.iterations = list()
        self.spread = list()  # {bid: float, ask: float}
        self.inventories = list()  # {trader: str, volume: float}

        # Miscellaneous
        self.bids = list()  # {volume: float, qty: int}
        self.asks = list()  # {volume: float, qty: int}
        self.states = list()  # {active: int, panic: int}

    def capture(self):
        order_book = self.market.order_book

        self.iterations.append(len(self.iterations))
        self.spread.append({'bid': order_book['bid'].first.price, 'ask': order_book['ask'].first.price})
        self.bids.append({'volume': sum([order.qty for order in order_book['bid']]), 'qty': len(order_book['bid'])})
        self.asks.append({'volume': sum([order.qty for order in order_book['ask']]), 'qty': len(order_book['ask'])})

        if self.market_makers:
            self.inventories.append({trader.name: trader.inventory for trader in self.market_makers})
            self.states.append(dict(Counter([trader.state for trader in self.market_makers])))

    # Market statistics
    # Numerical
    def center_price(self) -> np.ndarray:
        return (self.best_price('bid') + self.best_price('ask')) / 2

    def spread_size(self) -> np.ndarray:
        return self.best_price('ask') - self.best_price('bid')

    def best_price(self, order_type: str) -> np.ndarray:
        return np.array([spread[order_type] for spread in self.spread])

    def sum_quantity(self, order_type: str) -> np.ndarray:
        if order_type == 'bid':
            return np.ndarray([val['qty'] for val in self.bids])
        if order_type == 'ask':
            return np.ndarray([val['qty'] for val in self.asks])

    def sum_volume(self, order_type: str) -> np.ndarray:
        if order_type == 'bid':
            return np.array([val['volume'] for val in self.bids])
        if order_type == 'ask':
            return np.array([val['volume'] for val in self.asks])

    def excess_volume(self) -> np.ndarray:
        ask_volume = np.array([ask['volume'] for ask in self.asks])
        bid_volume = np.array([bid['volume'] for bid in self.bids])
        return ask_volume - bid_volume

    def price_diff(self) -> np.ndarray:
        return self.center_price()[1:] - self.center_price()[:-1]

    def volume_diff(self) -> np.ndarray:
        return self.excess_volume()[1:] - self.excess_volume()[:-1]

    def volatility(self, n=10) -> np.ndarray:
        price = self.center_price()
        return np.array([np.roll(price, -i)[:n].std() for i in range(price.shape[0])])

    # Market statistics
    # Categorical
    def price_states(self, k=10) -> np.ndarray:
        """
        *States:* increase (True), decrease (False)

        :param k: window size
        :return: list: [bool], True - increase, False - decrease
        """
        price_diff = self.price_diff()
        return np.array([np.roll(price_diff, -i)[:k].sum() > 0 for i in range(price_diff.shape[0])])

    def volume_states(self, k=10) -> np.ndarray:
        """
        *States:* increase (True), decrease (False)

        :param k: window size
        :return: list: [bool], True - increase, False - decrease
        """
        volume_diff = self.volume_diff()
        return np.array([np.roll(volume_diff, -i)[:k].sum() > 0 for i in range(volume_diff.shape[0])])

    def volatility_states(self, k=10) -> np.ndarray:
        """
        *States:* volatile (True), stable (False)

        :param k: window size
        :return: list: [bool], True - volatile, False - stable
        """
        volatility = self.volatility(n=k)
        if self.noise_agents:  # Here I determine threshold for volatility
            ### Maybe 1.96 * price_deviation
            price_deviation = 1 / self.noise_agents[0].lambda_  # Noise trader price standard deviation
        else:
            price_deviation = 1

        return volatility > price_deviation

    def market_states(self) -> np.ndarray:
        """
        *States:* price, volume - increasing or decreasing; volatility - volatile or stable
        """
        price = np.vectorize({True: 'i', False: 'd'}.get)(self.price_states(k=10))
        volume = np.vectorize({True: 'i', False: 'd'}.get)(self.volume_states(k=10))
        liquid = np.vectorize({True: 'vol', False: 'sta'}.get)(self.volatility_states(k=10))
        n = min([price.shape[0], volume.shape[0], liquid.shape[0]])
        states = [' '.join([price[i], volume[i], liquid[i]]) for i in range(n)]
        return np.array(states)

    @classmethod
    def states_markov(cls, states: list, order=None):
        """
        Turns list of records states for each iteration to the transition matrix between these states

        :param states: list of states records
        :param order: list of unique states by order
        :return: pandas.DataFrame of states transition matrix
        """
        result = dict()
        for i in range(1, len(states) - 1):
            p = states[i]
            n = states[i + 1]
            if not result.get(p):
                result[p] = dict()
            if not result[p].get(n):
                result[p][n] = 0
            result[p][n] += 1

        if not order:
            result = pd.DataFrame(result)
            result = result.sort_index()
            result = result[sorted(result.columns)]
        else:
            result = pd.DataFrame(result, columns=order, index=order).fillna(0)
        return result

    # Traders statistics
    def trader_inventory(self, trader: MarketMaker) -> list:
        return [inventory[trader.name] for inventory in self.inventories]

    def trader_states(self) -> list:
        """
        *States:* overflow, shortfall, balanced, panic

        :return:
        """
        states = list()
        for i, sample in enumerate(self.inventories):
            states.append(dict())
            for trader in self.market_makers:
                center = (trader.ul + trader.ll) / 2
                width = (trader.ul - trader.ll) / 2
                inventory = sample[trader.name]

                if abs(inventory - center) / width < .5:
                    states[i][trader.name] = 'balanced'
                elif abs(inventory - center) / width > .8:
                    states[i][trader.name] = 'panic'
                else:
                    if inventory < center:
                        states[i][trader.name] = 'shortfall'
                    else:
                        states[i][trader.name] = 'overflow'
        return states

    def panic_count(self):
        return [state['panic'] if state.get('panic') else 0 for state in self.states]

    def order_book_summary(self, order_type: str):
        return pd.DataFrame([*self.market.order_book[order_type].to_list()]).describe().round(2).transpose()
