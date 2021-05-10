from agents import ExchangeAgent, MarketMaker
from tqdm import tqdm
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from sklearn.linear_model import LinearRegression
import pymannkendall as mk


def mean(x: list) -> float:
    return sum(x) / len(x)


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
            plt.plot(self.info.iterations[10:], self.info.center_price()[10:], color='black', label='mean', lw=lw)
        else:
            plt.plot(self.info.iterations, lowess(self.info.center_price(), self.info.iterations, return_sorted=False),
                     color='black', label='mean', lw=lw)
        if show_spread:
            plt.plot(self.info.iterations, self.info.best_price('bid'), color='green', label='bid', lw=lw)
            plt.plot(self.info.iterations, self.info.best_price('bid'), color='red', label='ask', lw=lw)

        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_market_volume(self, lw=1, smoothing=False):
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

    def plot_states(self, lw=1):
        plt.plot(self.info.iterations, self.info.panic_count(), lw=lw, color='black')
        plt.title('Market Makers States')
        plt.ylabel('Panic')
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
        return [(spread['ask'] + spread['bid']) / 2 if spread['bid'] and spread['ask'] else 0 for spread in self.spread]

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
        result = list()
        for i in self.iterations:
            result.append(ask_volume[i] - bid_volume[i])
        return result

    def trader_inventory(self, trader: MarketMaker) -> list:
        return [inventory[trader.name] for inventory in self.inventories]

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
