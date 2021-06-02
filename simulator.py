from agents import ExchangeAgent, MarketMaker
from tqdm import tqdm
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from sklearn.linear_model import LinearRegression
import pymannkendall as mk


def mean(x: list) -> float:
    return sum(x) / len(x)


def diff(x: list) -> list:
    return [x[i + 1] - x[i] for i in range(len(x) - 1)]


def std(x: list) -> float:
    m = mean(x)
    return sum([(val - m)**2 for val in x])**0.5 / len(x)


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

    def fit(self, n, nt_lag=0, mm_lag=0):
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
                self.probe_trader.call()

        return self

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
        states = self.info.market_states()
        result = dict()
        for i in range(len(states) - 1):
            p = states[i]
            n = states[i + 1]
            if not result.get(p):
                result[p] = dict()
            if not result[p].get(n):
                result[p][n] = 0
            result[p][n] += 1

        result = pd.DataFrame(result)
        sns.heatmap(result, annot=True)
        plt.show()

    def test_trend(self, kpss_type='constant'):
        fuller = self.info.adfuller_price()
        kpss_inst = self.info.kpss_price(regression=kpss_type)

        print('Results of Dickey-Fuller Test:')
        dfoutput = pd.Series(fuller[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in fuller[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

        print('KPSS test:')
        kpss_output = pd.Series(kpss_inst[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
        for key, value in kpss_inst[3].items():
            kpss_output['Critical Value (%s)' % key] = value
        print(kpss_output)


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

    def center_price(self) -> list:
        bids = self.best_price('bid')
        asks = self.best_price('ask')
        return [(bids[i] + asks[i]) / 2 for i in range(len(self.iterations))]

    def spread_size(self) -> list:
        bids = self.best_price('bid')
        asks = self.best_price('ask')
        return [asks[i] - bids[i] for i in range(len(asks))]

    def best_price(self, order_type: str) -> list:
        return [spread[order_type] for spread in self.spread]

    def sum_quantity(self, order_type: str) -> list:
        if order_type == 'bid':
            return [val['qty'] for val in self.bids]
        if order_type == 'ask':
            return [val['qty'] for val in self.asks]

    def sum_volume(self, order_type: str) -> list:
        if order_type == 'bid':
            return [val['volume'] for val in self.bids]
        if order_type == 'ask':
            return [val['volume'] for val in self.asks]

    def excess_volume(self) -> list:
        ask_volume = [ask['volume'] for ask in self.asks]
        bid_volume = [bid['volume'] for bid in self.bids]
        return [ask_volume[i] - bid_volume[i] for i in range(len(ask_volume))]

    def price_diff(self) -> list:
        return diff(self.center_price())

    def volume_diff(self) -> list:
        return diff(self.excess_volume())

    def volatility(self, n=0) -> list:
        if not n:
            n = max(int(len(self.iterations) / 200), 10)

        x = self.center_price()
        vol = [0]
        for i in range(1, len(x)):
            if i - n > 0:
                vol.append(std(x[max(0, i - n):i]))
            else:
                vol.append(0)
        return vol

    def price_states(self, th: float, k=5) -> list:
        """
        *States:* increase, decrease, soar, fall

        :param th: threshold for fraction relative to the spread size
        :param k: number of steps for price
        :return: list of states (iterations - 1)
        """
        states = list()
        size = self.spread_size()
        price_diff = self.price_diff()

        for i in range(len(price_diff)):
            if i == 0:
                states.append('undefined')
                continue
            price_change = sum(price_diff[max(0, i - k):i])
            spread = size[0]

            if price_diff[i] > 0:
                if spread == 0:
                    states.append('soar')
                elif abs(price_change / spread) > th:
                    states.append('soar')
                else:
                    states.append('increase')
            else:
                if spread == 0:
                    states.append('fall')
                elif abs(price_change / spread) > th:
                    states.append('fall')
                else:
                    states.append('decrease')
        return states

    def volume_states(self, th, k=5) -> list:
        """
        *States:* increase, decrease, soar, fall

        :param th: threshold for fraction relative to the sum of volume on the opposite side of order book
        :param k: number of steps for price
        :return: list of states (iterations - 1)
        """
        states = list()
        size_ask = self.sum_volume('ask')
        size_bid = self.sum_volume('bid')
        volume_diff = self.volume_diff()

        for i in range(len(volume_diff)):
            volume_change = sum(volume_diff[max(0, i - k):i])

            if volume_diff[i] > 0:
                if abs(volume_diff[i] / size_bid[i]) > th:
                    states.append('soar')
                else:
                    states.append('increase')
            else:
                if abs(volume_diff[i] / size_ask[i]) > th:
                    states.append('fall')
                else:
                    states.append('decrease')
        return states

    def volatility_states(self, th, n=0) -> list:
        """
        *States:* volatile, static

        :param th:
        :param n:
        :return:
        """
        vol = self.volatility()
        states = list()
        for i in range(len(vol)):
            if vol[i] > th:
                states.append('volatile')
            else:
                states.append('static')
        return states

    def market_states(self) -> list:
        """
        *States:* dangerous, volatile, shocked, balanced

        :return:
        """
        price = self.price_states(.7)
        volume = self.volume_states(.7)
        liquid = self.volume_states(.3)
        states = list()

        for i in range(len(price)):
            if liquid[i] == 'volatile':
                if (price[i] == 'soar' or price[i] == 'fall') or (volume[i] == 'soar' or volume[i] == 'fall'):
                    states.append('dangerous')
                else:
                    states.append('volatile')
            else:
                if (price[i] == 'soar' or price[i] == 'fall') or (volume[i] == 'soar' or volume[i] == 'fall'):
                    states.append('shocked')
                else:
                    states.append('balanced')
        return states

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

    def adfuller_price(self, metric='AIC') -> adfuller:
        """
        The Augmented Dickey-Fuller test can be used to test for a unit root in a univariate process in the
        presence of serial correlation. H0 - non-stationary, H1 - stationary

        :param metric: {“AIC”, “BIC”, “t-stat”, None}
        :return: statistic, p-value, lags, nobs, critical values
        """
        return adfuller(self.center_price(), autolag=metric)

    def kpss_price(self, regression='constant', lags='auto') -> kpss:
        """
        Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null hypothesis that x is level or trend
        stationary. H0 - trend is stationary, H1 - trend is non-stationary

        :param regression: const - the data is stationary around a constant, trend - the data is stationary around the
        trend
        :param lags
        """
        regression = {'constant': 'c', 'trend': 'ct'}[regression]
        return kpss(self.center_price(), regression=regression, nlags=lags)

    def mann_kendall_price(self):
        """
        The Mann Kendall Trend Test (sometimes called the M-K test) is used to analyze data collected over time for
        consistently increasing or decreasing trends (monotonic) in Y values. H0 - there is no monotonic trend,
        H1 - the trend exists, it is either positive or negative

        :return: trend: trend direction, h: bool if trend exists, p: p-value, z: z-stat, Tau: Kendall Tau,
        s: Mann-Kendal’s score, var_s: Variance S, slope: Theil-Sen estimator/slope,
        intercept: Intercept of Kendall-Theil Robust Line
        """
        return mk.original_test(self.center_price())

    def t_test_price(self):
        x = np.array(self.iterations).reshape(-1, 1)
        y = np.array(self.center_price()).reshape(-1, 1)

        lm = LinearRegression().fit(x, y)
        params = np.append(lm.intercept_, lm.coef_)
        predictions = lm.predict(x)

        newX = pd.DataFrame({"Constant": np.ones(len(x))}).join(pd.DataFrame(x))
        MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

        # Note if you don't want to use a DataFrame replace the two lines above with
        # newX = np.append(np.ones((len(X),1)), X, axis=1)
        # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b

        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]

        sd_b = np.round(sd_b, 3)
        ts_b = np.round(ts_b, 3)
        p_values = np.round(p_values, 3)
        params = np.round(params, 4)
        return {'params': params, 't': ts_b, 'sd': sd_b, 'p-value': p_values}

    def order_book_summary(self, order_type: str):
        bid = pd.DataFrame([*self.market.order_book['bid'].to_list()])
        ask = pd.DataFrame([*self.market.order_book['ask'].to_list()])
        return ['bid', bid.describe().round(2).transpose(), 'ask', ask.describe().round(2).transpose()]

