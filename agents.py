import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque


class Order:
    def __init__(self, price, qty, order_type, trader_link=None):
        # Properties
        self.price = price
        self.qty = qty
        self.order_type = order_type
        self.trader = trader_link
        # Connections
        self.left = None
        self.right = None

    def to_dict(self):
        return {'price': self.price, 'qty': self.qty, 'order_type': self.order_type,
                'trader_link': self.trader}  # trader.name


class OrderIter:
    def __init__(self, order_list):
        self.order = order_list.first

    def __next__(self):
        if self.order:
            next_order = self.order
            self.order = self.order.right
            return next_order
        raise StopIteration


class OrderList:
    def __init__(self, order_type: str):
        self.first = None
        self.last = None
        self.order_type = order_type

    def __iter__(self):
        return OrderIter(self)

    def to_list(self):
        return [order.to_dict() for order in self]

    def remove(self, order: Order):
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')
        if order == self.first:
            self.first = order.right
        if order == self.last:
            self.last = order.left

        if order.left:
            order.left.right = order.right
        if order.right:
            order.right.left = order.left

    def append(self, order: Order):
        if not self.first:
            self.first = order
            self.last = order
            return

        self.last.right = order
        order.left = self.last
        self.last = order

    def insert(self, order: Order):
        # If wrong order type to insert
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        # If empty
        if not self.first:
            self.append(order)
            return

        if self.order_type == 'bid':
            # Insert order in the beginning
            if order.price >= self.first.price:
                order.right = self.first
                self.first.left = order
                self.first = order
                return

            # Insert order in the middle
            for val in self:
                if order.price >= val.price:
                    order.left = val.left
                    order.right = val
                    order.left.right = order
                    order.right.left = order
                    return
        else:
            for val in self:
                # Insert order in the beginning
                if order.price <= self.first.price:
                    order.right = self.first
                    self.first.left = order
                    self.first = order
                    return

                # Insert order in the middle
                if order.price <= val.price:
                    order.left = val.left
                    order.right = val
                    order.left.right = order
                    order.right.left = order
                    return

        # Insert to the end
        self.append(order)

    def fulfill(self, order: Order) -> Order:
        if order.order_type == self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        if self.order_type == 'bid':
            for val in self:
                if order.qty == 0:
                    break
                if val.price < order.price:
                    break

                tmp = min(order.qty, val.qty)  # Quantity traded currently
                val.qty -= tmp
                order.qty -= tmp

                if val.qty == 0:
                    self.remove(val)

        else:
            for val in self:
                if order.qty == 0:
                    break
                if val.price > order.price:
                    break

                tmp = min(order.qty, val.qty)  # Quantity traded currently
                val.qty -= tmp
                order.qty -= tmp

                if val.qty == 0:
                    self.remove(val)

        return order

    def from_list(self, order_list, sort=False):
        order_list = [Order(order['price'], order['qty'], order['order_type'],
                            order.get('trader_link')) for order in order_list]
        if sort:
            for order in order_list:
                self.insert(order)
        else:
            for order in order_list:
                self.append(order)


class ExchangeAgent:
    def __init__(self, spread_init=None, depth=0, lambda_=2, mu=0, sigma=1):
        self.order_book = {'bid': deque(), 'ask': deque()}
        self.name = 'market'
        self.inventory = 0

        # Initialize spread
        if spread_init:
            self.order_book['bid'].append({'price': spread_init['bid'], 'qty': 1, 'agent': self})
            self.order_book['ask'].append({'price': spread_init['ask'], 'qty': 1, 'agent': self})

        # Add noise limit orders
        for i in tqdm(range(depth), desc='Market Initialization'):
            delta = random.expovariate(lambda_)
            quantity = random.lognormvariate(mu, sigma)

            # BID
            self.limit_order('bid', quantity, spread_init['bid'] - delta, self)
            # ASK
            self.limit_order('ask', quantity, spread_init['ask'] + delta, self)

    def _clear_glass(self, order_type):
        if order_type == 'bid':
            self.order_book['ask'] = deque([order for order in self.order_book['ask'] if order['qty'] > 0])
        if order_type == 'ask':
            self.order_book['bid'] = deque([order for order in self.order_book['bid'] if order['qty'] > 0])

    def spread(self):
        if not self.order_book['bid']:
            bid = None
        else:
            bid = self.order_book['bid'][0]['price']

        if not self.order_book['ask']:
            ask = None
        else:
            ask = self.order_book['ask'][0]['price']

        if bid is None and ask is None:
            return None
        return {'bid': bid, 'ask': ask}

    def spread_volume(self):
        if not self.order_book['bid']:
            bid = None
        else:
            bid = self.order_book['bid'][0]['qty']

        if not self.order_book['ask']:
            ask = None
        else:
            ask = self.order_book['ask'][0]['qty']

        if bid is None and ask is None:
            return None
        return {'bid': bid, 'ask': ask}

    def limit_order(self, order_type, quantity, price, trader_link):
        """
        Executes limit order.

        If inside spread and spread exist -> add order in the beginning

        For BID:
        If lower than spread:
        1) Iterate over BID orders (decreasing) until OUR price is greater than the BID

        If upper than spread:
        1) Iterate over ASK orders until OUR price is lower than the ASK price
        or until the order is fulfilled
        2) If the order is left unfilled, add it in the beginning of the BID list
        3) clear glass from resolved orders

        For ASK:
        If upper than spread:
        1) Iterate over ASK orders (increasing) until OUR price is lower than the ASK

        If lower than spread:
        1) Iterate over ASK orders until OUR price is lower than the ASK price
        or until the order is fulfilled
        2) If the order is left unfilled, add it in the beginning of the ASK list
        3) clear glass from resolved orders

        :return: void
        """

        spread = self.spread()
        # If glass is empty
        if spread is None:
            self.order_book[order_type].append({'price': price, 'qty': quantity, 'agent': trader_link})
            return

        # If inside spread
        if spread is not None and spread['bid'] is not None and spread['ask'] is not None:
            if spread['bid'] < price < spread['ask']:
                self.order_book[order_type].insert(0, {'price': price, 'qty': quantity, 'agent': trader_link})
                return

        # Raise if spread is negative
        if spread['bid'] is not None and spread['ask'] is not None:
            if spread['bid'] >= spread['ask']:
                raise ValueError('Spread is negative')

        # If BID
        if order_type == 'bid':

            # If BID is empty
            if spread['bid'] is None and price < spread['ask']:
                self.order_book['bid'].append({'price': price, 'qty': quantity, 'agent': trader_link})
                return

            # If in BID side of glass
            if spread['bid'] is not None and price <= spread['bid']:
                for i, order in enumerate(self.order_book['bid']):
                    if price == order['price']:
                        self.order_book['bid'][i]['qty'] += quantity
                        return
                    if price > order['price']:
                        self.order_book['bid'].insert(i, {'price': price, 'qty': quantity, 'agent': trader_link})
                        return
                self.order_book['bid'].append({'price': price, 'qty': quantity, 'agent': trader_link})
                return

            # If in ASK side of glass
            if spread['ask'] is None or price >= spread['ask']:
                for i, order in enumerate(self.order_book['ask']):
                    if quantity == 0:
                        break
                    if price < order['price']:
                        break

                    tmp = min(quantity, order['qty'])  # Quantity traded currently
                    self.order_book['ask'][i]['qty'] -= tmp
                    self.order_book['ask'][i]['agent'].inventory -= tmp  # Decrease inventory of seller
                    trader_link.inventory += tmp  # Increase inventory of buyer
                    quantity -= tmp

                if quantity > 0:
                    self.order_book['bid'].insert(0, {'price': price, 'qty': quantity, 'agent': trader_link})
                self._clear_glass(order_type)  # Clear glass from resolved orders
                return

        # If ASK
        if order_type == 'ask':

            # If ASK is empty
            if spread['ask'] is None and price > spread['bid']:
                self.order_book['ask'].append({'price': price, 'qty': quantity, 'agent': trader_link})
                return

            # If in ASK side of glass
            if spread['ask'] is not None and price >= spread['ask']:
                for i, order in enumerate(self.order_book['ask']):
                    if price == order['price']:
                        self.order_book['ask'][i]['qty'] += quantity
                        return
                    if price < order['price']:
                        self.order_book['ask'].insert(i, {'price': price, 'qty': quantity, 'agent': trader_link})
                        return
                self.order_book['ask'].append({'price': price, 'qty': quantity, 'agent': trader_link})
                return

            # If in BID side of glass
            if spread['bid'] is None or price <= spread['bid']:
                for i, order in enumerate(self.order_book['bid']):
                    if quantity == 0:
                        break
                    if price > order['price']:
                        break

                    tmp = min(quantity, order['qty'])  # Quantity trader currently
                    self.order_book['bid'][i]['qty'] -= tmp
                    self.order_book['bid'][i]['agent'].inventory += tmp  # Increase inventory of buyer
                    trader_link.inventory -= tmp  # Decrease inventory of seller
                    quantity -= tmp

                if quantity > 0:
                    self.order_book['ask'].insert(0, {'price': price, 'qty': quantity, 'agent': trader_link})
                self._clear_glass(order_type)  # Clear glass from resolved orders
                return

    def market_order(self, order_type, quantity, trader_link):
        """
        Executes market order

        For BID:
        1) Iterate through ASK and fulfill orders if OUR quantity is 0 then stop and return TRUE
        2) If quantity > 0 then return False

        For ASK:
        1) Iterate through BID and fulfill orders if OUR quantity is 0 then stop and return TRUE
        2) If quantity > 0 then return False

        :return: qty remaining
        """

        # If BID
        if order_type == 'bid':
            for i, order in enumerate(self.order_book['ask']):
                if quantity == 0:
                    break

                tmp = min(quantity, order['qty'])  # Quantity trader currently
                self.order_book['ask'][i]['qty'] -= tmp
                self.order_book['ask'][i]['agent'].inventory -= tmp  # Decrease inventory of seller
                trader_link.inventory += tmp  # Increase inventory of buyer
                quantity -= tmp

            self._clear_glass(order_type)  # Clear glass from resolved orders
            return quantity

        # If ASK
        if order_type == 'ask':
            for i, order in enumerate(self.order_book['bid']):
                if quantity == 0:
                    break

                tmp = min(quantity, order['qty'])  # Quantity trader currently
                self.order_book['bid'][i]['qty'] -= tmp
                self.order_book['bid'][i]['agent'].inventory += tmp  # Increase inventory of buyer
                trader_link.inventory -= tmp  # Decrease inventory of seller
                quantity -= tmp

            self._clear_glass(order_type)  # Clear glass from resolved orders
            return quantity

    def cancel_order(self, order_type, trader_link):
        """
        Cancel random order from the order_type side of the order book

        :return: bool (True - order removed, False - otherwise)
        """
        for i, order in enumerate(self.order_book[order_type]):
            if order['agent'] == trader_link:
                del self.order_book[order_type][i]
                return True
        return False

    def cancel_all(self, order_type, trader_link):
        if order_type == 'bid':
            self.order_book['ask'] = deque([order for order in self.order_book['ask']
                                           if order['agent'].name != trader_link.name])
        if order_type == 'ask':
            self.order_book['bid'] = deque([order for order in self.order_book['bid']
                                           if order['agent'].name != trader_link.name])

    def plot_order_book(self):
        ax = plt.subplot()

        for order_type in ('bid', 'ask'):
            qtys = [order['qty'] for order in self.order_book[order_type]]
            prices = [order['price'] for order in self.order_book[order_type]]
            color = 'green' if order_type == 'bid' else 'red'
            ax.scatter(prices, qtys, label=order_type, c=color, s=5)

        ax.vlines(self.spread()['bid'], 0, max(qtys), color='black', ls='--', alpha=.8)
        ax.vlines(self.spread()['ask'], 0, max(qtys), color='black', ls='--', alpha=.8)
        ax.legend()
        ax.set_title('Order Book')


class Trader:
    def __init__(self, market: ExchangeAgent, agent_id, inventory, agent_name='Undefined'):
        self.market = market
        self.name = f'{agent_name}{agent_id}'
        self.inventory = inventory

    def _buy_limit(self, quantity, price):
        self.market.limit_order('bid', quantity, price, self)

    def _sell_limit(self, quantity, price):
        self.market.limit_order('ask', quantity, price, self)

    def _buy_market(self, quantity):
        self.market.market_order('bid', quantity, self)

    def _sell_market(self, quantity):
        self.market.market_order('ask', quantity, self)

    def _cancel_order(self, order_type):
        self.market.cancel_order(order_type, self)


class NoiseAgent(Trader):
    def __init__(self, market: ExchangeAgent, agent_id, inventory, lambda_=2, mu=0, sigma=1):
        super().__init__(market, agent_id, inventory, agent_name='NoiseTrader')
        self.lambda_ = lambda_
        self.mu = mu
        self.sigma = sigma

    def _draw_price(self, order_type, spread, lambda_):
        """
        Draw price for limit order of Noise Agent. The price is calculated as:
        1) 35% - within the spread - uniform distribution
        2) 65% - out of the spread - delta from best price is exponential distribution r.v.

        :param order_type: 'bid' or 'ask'
        :param spread:
        :param lambda_: from 0.1 for big markets to 2 for small markets
        :return: price
        """
        random_state = random.random()  # Determines IN spread OR OUT of spread

        # Within the spread
        if random_state < .35:
            return random.uniform(spread['bid'], spread['ask'])

        # Out of spread
        else:
            delta = random.expovariate(lambda_)
            if order_type == 'bid':
                return spread['bid'] - delta
            if order_type == 'ask':
                return spread['ask'] + delta

    def _draw_quantity(self, order_type, order_exec, mu, sigma):
        """
        Draw quantity for any order of Noise Agent.
        1) If market order - volume is the volume of the best offer
        2) If limit order - volume is derived from log-normal distrivution

        :param order_type: 'bid' or 'ask'
        :param order_exec: 'market' or 'limit'
        :param mu: parameter of log-normal distribution
        :param sigma: parameter of log-normal distribution
        :return: quantity for order
        """
        spread_volume = self.market.spread_volume()

        # Market order
        if order_exec == 'market':
            return random.lognormvariate(mu, sigma)

        # Limit order
        if order_exec == 'limit':
            return random.lognormvariate(mu, sigma)

    def call(self, spread, random_state=None):
        """
        Function to call agent action
        :param spread:
        :param random_state: random unif (0,1) which prescribes which action to perform
        :return: void
        """
        if not spread['bid'] or not spread['ask']:
            return

        if random_state is None:
            random_state = random.random()

        if random_state > .5:
            order_type = 'bid'
        else:
            order_type = 'ask'

        random_state = random.random()
        # Market order
        if random_state > .85:
            quantity = self._draw_quantity(order_type, 'market', self.mu, self.sigma)
            if order_type == 'bid':
                self._buy_market(quantity)
            else:
                self._sell_market(quantity)

        # Limit order
        elif random_state > .50:
            price = self._draw_price(order_type, spread, self.lambda_)
            quantity = self._draw_quantity(order_type, 'limit', self.mu, self.sigma)
            if order_type == 'bid':
                self._buy_limit(quantity, price)
            else:
                self._sell_limit(quantity, price)

        # Cancellation order
        elif random_state < .5:
            self._cancel_order(order_type)


class MarketMaker(Trader):
    def __init__(self, market: ExchangeAgent, agent_id, inventory, ul, ll):
        """
        self.state in (Active, Panic)
        """
        super().__init__(market, agent_id, inventory, agent_name='MarketMaker')
        self.ul = ul  # Upper Limit
        self.ll = ll  # Lower Limit
        self.state = None

    def call(self, spread):
        if not spread['bid'] or not spread['ask']:
            self.state = 'Panic'
            return

        # Clear previous orders
        self.market.cancel_all('bid', self)
        self.market.cancel_all('ask', self)

        bid_volume = max(0, self.ul - 1 - self.inventory)
        ask_volume = max(0, self.inventory - self.ll - 1)

        if not bid_volume or not ask_volume:
            self.state = 'Panic'
        else:
            self.state = 'Active'

        base_offset = -(spread['ask'] - spread['bid'] - 1) * (self.inventory / self.ul)

        # BID
        self._buy_limit(bid_volume, spread['bid'] + base_offset)
        # ASK
        self._sell_limit(ask_volume, spread['ask'] + base_offset)
