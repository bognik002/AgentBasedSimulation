import random
from tqdm import tqdm


class Order:
    """
    Order contains all relevant information about order, it can be of two types: bid, ask. Supports binary comparison
    operations of price among all pairs of order types according to the logic:

    **better-offer < worse-offer**
    """
    order_id = 0

    def __init__(self, price, qty, order_type, trader_link=None):
        # Properties
        self.price = price
        self.qty = qty
        self.order_type = order_type
        self.trader = trader_link
        self.order_id = Order.order_id
        # Connections
        self.left = None
        self.right = None
        Order.order_id += 1

    def __lt__(self, other) -> bool:
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price > other.price
            if self.order_type == 'ask':
                return self.price < other.price

        if self.order_type == 'bid':
            return self.price > other.price
        if self.order_type == 'ask':
            return self.price < other.price

    def __le__(self, other):
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price >= other.price
            if self.order_type == 'ask':
                return self.price <= other.price

        if self.order_type == 'bid':
            return self.price >= other.price
        if self.order_type == 'ask':
            return self.price <= other.price

    def __gt__(self, other):
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price < other.price
            if self.order_type == 'ask':
                return self.price > other.price

        if self.order_type == 'bid':
            return self.price < other.price
        if self.order_type == 'ask':
            return self.price > other.price

    def __ge__(self, other):
        if self.order_type != other.order_type:
            if self.order_type == 'bid':
                return self.price <= other.price
            if self.order_type == 'ask':
                return self.price >= other.price

        if self.order_type == 'bid':
            return self.price <= other.price
        if self.order_type == 'ask':
            return self.price >= other.price

    def __str__(self):
        return f'Order{self.order_id} {self.order_type}'

    def to_dict(self) -> dict:
        return {'price': self.price, 'qty': self.qty, 'order_type': self.order_type,
                'trader_link': self.trader}

    @classmethod
    def from_dict(cls, order_dict):
        return Order(order_dict['price'], order_dict['qty'], order_dict['order_type'], order_dict.get('trader_link'))


class OrderIter:
    """
    Iterator class for OrderList
    """
    def __init__(self, order_list):
        self.order = order_list.first

    def __next__(self) -> Order:
        if self.order:
            next_order = self.order
            self.order = self.order.right
            return next_order
        raise StopIteration


# noinspection DuplicatedCode
class OrderList:
    """
    OrderList is implemented as a doubly linked list. It preserves the same order type inside,
    all orders are sorted according to best-offer -> worst-offer dynamically.

    remove, append, push: complexity O(1)

    insert, fulfill (for large qty): complexity O(n)
    """
    def __init__(self, order_type: str):
        self.first = None
        self.last = None
        self.order_type = order_type

    def __iter__(self) -> OrderIter:
        return OrderIter(self)

    def __bool__(self):
        return self.first is not None and self.last is not None

    def __len__(self):
        n = 0
        for order in self:
            n += 1
        return n

    def to_list(self) -> list:
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

        order.left = None
        order.right = None

    def append(self, order: Order):
        # If wrong order type to insert
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        if not self.first:
            self.first = order
            self.last = order
            return

        self.last.right = order
        order.left = self.last
        self.last = order

    def push(self, order: Order):
        """
        Insert order in the beginning

        :param order: Order
        :return: void
        """
        # If wrong order type to insert
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        if not self.first:
            self.first = order
            self.last = order
            return

        self.first.left = order
        order.right = self.first
        self.first = order

        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

    def insert(self, order: Order):
        """
        Inserts order preserving best-offer -> worst-offer

        :param order: Order
        :return: void
        """
        # If wrong order type to insert
        if order.order_type != self.order_type:
            raise ValueError(f'Wrong order type! OrderList: {self.order_type}, Order: {order.order_type}')

        # Insert order in the beginning
        if order <= self.first:
            order.right = self.first
            self.first.left = order
            self.first = order
            return

        # Insert order in the middle
        for val in self:
            if order <= val:
                # If only 1 order in self
                if self.first == self.last:
                    self.push(order)

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

        for val in self:
            if order.qty == 0:
                break
            if val > order:
                break

            tmp = min(order.qty, val.qty)  # Quantity traded currently
            val.qty -= tmp
            order.qty -= tmp

            if val.qty == 0:
                self.remove(val)

        return order

    @classmethod
    def from_list(cls, order_list, sort=False):
        order_list = [Order(order['price'], order['qty'], order['order_type'],
                            order.get('trader_link')) for order in order_list]
        order_list_obj = OrderList(order_list[0].order_type)
        if sort:
            for order in order_list:
                order_list_obj.insert(order)
        else:
            for order in order_list:
                order_list_obj.append(order)
        return order_list_obj


class ExchangeAgent:
    """
    ExchangeAgent implements automatic orders handling within the order book. It supports limit orders,
    market orders, cancel orders, returns current spread prices and volumes.
    """
    def __init__(self, spread_init, depth=0, price_std=3, quantity_mean=0, quantity_std=1):
        self.order_book = {'bid': OrderList('bid'), 'ask': OrderList('ask')}
        self.name = 'market'

        # Initialize spread
        if spread_init:
            self.order_book['bid'].append(Order.from_dict({'price': spread_init['bid'], 'qty': 1, 'trader_link': self,
                                                           'order_type': 'bid'}))
            self.order_book['ask'].append(Order.from_dict({'price': spread_init['ask'], 'qty': 1, 'trader_link': self,
                                                           'order_type': 'ask'}))

        # Add noise limit orders
        for i in tqdm(range(depth), desc='Market Initialization'):
            delta = random.expovariate(1 / price_std)
            quantity = random.lognormvariate(quantity_mean, quantity_std)

            self.limit_order(Order(spread_init['bid'] - delta, quantity, 'bid', self))  # BID
            self.limit_order(Order(spread_init['ask'] + delta, quantity, 'ask', self))  # ASK

    def _clear_glass(self):
        """
        Clears glass from orders with 0 qty.

        complexity O(n)

        :return: void
        """
        self.order_book['bid'] = OrderList.from_list([order for order in self.order_book['bid'] if order.qty > 0])
        self.order_book['ask'] = OrderList.from_list([order for order in self.order_book['ask'] if order.qty > 0])

    def spread(self) -> dict:
        """
        :return: {'bid': float, 'ask': float}
        """
        if not self.order_book['bid'] or not self.order_book['ask']:
            return {'bid': None, 'ask': None}
        return {'bid': self.order_book['bid'].first.price, 'ask': self.order_book['ask'].first.price}

    def spread_volume(self) -> dict:
        """
        :return: {'bid': float, 'ask': float}
        """
        return {'bid': self.order_book['bid'].first.qty, 'ask': self.order_book['ask'].first.qty}

    def limit_order(self, order: Order):
        """
        Executes limit order, fulfilling orders if on other side of spread

        :return: void
        """
        bid, ask = self.spread().values()
        if not bid or not ask:
            return

        if order.order_type == 'bid':
            if order.price >= ask:
                order = self.order_book['ask'].fulfill(order)
            if order.qty > 0:
                self.order_book['bid'].insert(order)
            return

        elif order.order_type == 'ask':
            if order.price <= bid:
                order = self.order_book['bid'].fulfill(order)
            if order.qty > 0:
                self.order_book['ask'].insert(order)

    def market_order(self, order: Order) -> Order:
        """
        Executes market order, fulfilling orders on the other side of spread

        :return: Order
        """
        if order.order_type == 'bid':
            order = self.order_book['ask'].fulfill(order)
        elif order.order_type == 'ask':
            order = self.order_book['bid'].fulfill(order)
        return order

    def cancel_order(self, order: Order):
        """
        Cancel order from order book

        :return: void
        """
        if order.order_type == 'bid':
            self.order_book['bid'].remove(order)
        elif order.order_type == 'ask':
            self.order_book['ask'].remove(order)


class Trader:
    def __init__(self, market: ExchangeAgent):
        self.market = market
        self.orders = list()
        self.name = 'Undefined'

    def __str__(self) -> str:
        return self.name

    def _buy_limit(self, quantity, price):
        order = Order(price, quantity, 'bid', self)
        self.orders.append(order)
        self.market.limit_order(order)

    def _sell_limit(self, quantity, price):
        order = Order(price, quantity, 'ask', self)
        self.orders.append(order)
        self.market.limit_order(order)

    def _buy_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if not self.market.order_book['ask']:
            return quantity
        order = Order(self.market.order_book['ask'].last.price, quantity, 'bid', self)
        return self.market.market_order(order).qty

    def _sell_market(self, quantity) -> int:
        """
        :return: quantity unfulfilled
        """
        if not self.market.order_book['bid']:
            return quantity
        order = Order(self.market.order_book['bid'].last.price, quantity, 'ask', self)
        return self.market.market_order(order).qty

    def _cancel_order(self, order: Order):
        self.market.cancel_order(order)
        self.orders.remove(order)


class NoiseAgent(Trader):
    """
    NoiseTrader perform action on call. Creates noisy orders to recreate trading in real environment.
    """
    trader_id = 0

    def __init__(self, market: ExchangeAgent, price_std=2, quantity_mean=0, quantity_std=1):
        super().__init__(market)
        self.name = f'NoiseTrader{self.trader_id}'
        NoiseAgent.trader_id += 1

        self.lambda_ = 1 / price_std
        self.mu = quantity_mean
        self.sigma = quantity_std

    def _draw_price(self, order_type, spread: dict) -> float:
        """
        Draw price for limit order of Noise Agent. The price is calculated as:
        1) 35% - within the spread - uniform distribution
        2) 65% - out of the spread - delta from best price is exponential distribution r.v.

        :return: price
        """
        random_state = random.random()  # Determines IN spread OR OUT of spread

        # Within the spread
        if random_state < .35:
            return random.uniform(spread['bid'], spread['ask'])

        # Out of spread
        else:
            delta = random.expovariate(self.lambda_)
            if order_type == 'bid':
                return spread['bid'] - delta
            if order_type == 'ask':
                return spread['ask'] + delta

    def _draw_quantity(self, order_exec) -> float:
        """
        Draw quantity for any order of Noise Agent.
        1) If market order - currently the same as for limit order
        2) If limit order - volume is derived from log-normal distribution

        :param order_exec: 'market' or 'limit'
        :return: quantity for order
        """
        # Market order
        if order_exec == 'market':
            return random.lognormvariate(self.mu, self.sigma)

        # Limit order
        if order_exec == 'limit':
            return random.lognormvariate(self.mu, self.sigma)

    def call(self, spread: dict):
        """
        Function to call agent action

        :param spread: {'bid': float, 'ask' float} - spread stamp with lag
        :return: void
        """
        if not spread['bid'] or not spread['ask']:
            return

        random_state = random.random()

        if random_state > .5:
            order_type = 'bid'
        else:
            order_type = 'ask'

        random_state = random.random()
        # Market order
        if random_state > .85:
            quantity = self._draw_quantity('market')
            if order_type == 'bid':
                self._buy_market(quantity)
            elif order_type == 'ask':
                self._sell_market(quantity)

        # Limit order
        elif random_state > .50:
            price = self._draw_price(order_type, spread)
            quantity = self._draw_quantity('limit')
            if order_type == 'bid':
                self._buy_limit(quantity, price)
            elif order_type == 'ask':
                self._sell_limit(quantity, price)

        # Cancellation order
        elif random_state < .5:
            if self.orders:
                order_n = random.randint(0, len(self.orders) - 1)
                self._cancel_order(self.orders[order_n])


class MarketMaker(Trader):
    """
    MarketMaker performs action on call. Creates orders on either side of bid-ask spread, each time cancelling
    orders from previous call. Has inventory, which is recalculated after each trading session, and states
    (Active, Panic) which are determined by inventory left after trading session.
    """
    trader_id = 0

    def __init__(self, market: ExchangeAgent, inventory: float, upper_limit: float, lower_limit: float):
        super().__init__(market)
        self.name = f'MarketMaker{self.trader_id}'
        MarketMaker.trader_id += 1
        self.ul = upper_limit  # Upper Limit
        self.ll = lower_limit  # Lower Limit
        self.inventory = inventory
        self._trading_volume = {'bid': 0, 'ask': 0}
        self.state = 'Active'

    def _cancel_all(self):
        """
        Clear all previous orders and update inventory

        :return: void
        """
        left = {'bid': 0, 'ask': 0}  # Left unfulfilled
        for order in self.orders:
            left[order.order_type] += order.qty
            self._cancel_order(order)
        # Update inventory
        self.inventory += self._trading_volume['bid'] - left['bid']
        self.inventory -= self._trading_volume['ask'] - left['ask']

        self._trading_volume = {'bid': 0, 'ask': 0}  # Update trading volume

    def call(self, spread: dict):
        """
        Function to call MarketMaker action

        :param spread: {'bid': float, 'ask' float} - spread stamp with lag
        :return: void
        """
        # If market run out of orders on either side -> Panic
        if not spread['bid'] or not spread['ask']:
            self.state = 'Panic'
            return

        # Clear previous orders
        self._cancel_all()

        # Calculate bid and ask volume
        bid_volume = max(0., self.ul - 1 - self.inventory)
        ask_volume = max(0., self.inventory - self.ll - 1)
        self._trading_volume = {'bid': bid_volume, 'ask': ask_volume}  # Control how much is being traded

        # If in panic state we only either sell or buy commodities
        if not bid_volume or not ask_volume:
            self.state = 'Panic'
        else:
            self.state = 'Active'

        base_offset = -(spread['ask'] - spread['bid'] - 1) * (self.inventory / self.ul)  # Price offset

        self._buy_limit(bid_volume, spread['bid'] - base_offset)  # BID
        self._sell_limit(ask_volume, spread['ask'] + base_offset)  # ASK
