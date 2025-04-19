from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import math
import jsonpickle

import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:

    def mac_mprice(self, state: TradingState):
        product = "MAGNIFICENT_MACARONS"

        buy_ord = state.order_depths[product].buy_orders
        sell_ord = state.order_depths[product].sell_orders

        mid_weight_curr = 0
        total_weight = 0

        # fairprice = int((b_max * buy_ord[b_max] - s_min * sell_ord[s_min])/(buy_ord[b_max] - sell_ord[s_min]))
        for p in buy_ord:
            if buy_ord[p] > 0:
                mid_weight_curr += p * buy_ord[p]
                total_weight += buy_ord[p]

        for p in sell_ord:
            if -sell_ord[p] > 0:
                mid_weight_curr -= p * sell_ord[p]
                total_weight -= sell_ord[p]

        mid_price = mid_weight_curr / total_weight if total_weight > 0 else 0

        return mid_price
    
    def conv_ord(self, state):
        product = "MAGNIFICENT_MACARONS"
        conv_limit = 10

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        # Define available conversions
        if position > 0:
            conv_num = min(conv_limit, position)
        else:
            conv_num = 0

        return conv_num

    def mac_ord(self, state: TradingState):
        product = "MAGNIFICENT_MACARONS"
        pos_limit = 75
        conv_limit = 10
        margin_take = 1
        margin_put = 2
        margin_buy = 1
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        # Get observation parameters from the state
        observation = state.observations.conversionObservations[product]

        implied_bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        implied_ask = observation.askPrice + observation.importTariff + observation.transportFees

        # Find fair price
        midprice = self.mac_mprice(state)
        fairprice = implied_ask

        buy_ord = state.order_depths[product].buy_orders
        sell_ord = state.order_depths[product].sell_orders
        
        # always positive quantities indicating changes in the current position
        buy_quantity = 0
        sell_quantity = 0

        # Market sell if there are bids not far away from implied_ask
        # bid_highest = max(list(buy_ord.keys()))
        # if bid_highest >= implied_ask - 1 and pos_limit + position - sell_quantity > 0:
        #         sell_amount = min(buy_ord[bid_highest], pos_limit + position - sell_quantity,conv_limit)
        #         orders.append(Order(product,bid_highest,-sell_amount))
        #         sell_quantity += sell_amount
        # conv_num = sell_amount

        # If there are asks below observation asks plus margin, buy them
        # for p in sorted(list(sell_ord.keys())):
        #     if p <= implied_ask + margin_take and pos_limit - position - buy_quantity > 0:
        #         buy_amount = min(-sell_ord[p], pos_limit - position - buy_quantity)
        #         orders.append(Order(product,p,buy_amount))
        #         buy_quantity += buy_amount
        #         if buy_amount < -sell_ord[p]:
        #             ask_lowest = p
        #             break
        #     else:
        #         ask_lowest = p
        #         break

        # When available positions were closed, put out Orders at maximum profit
        ask_lowest = min(list(sell_ord.keys()))
        ask_amount = pos_limit + position - sell_quantity
        if ask_lowest >= int(fairprice) + 1 and ask_amount > 0:
            orders.append(Order(product, math.floor(fairprice),-ask_amount))
            logger.print(f"Sell orders posted at fair price {orders}")
        else:
            logger.print(f"No order: ask amount is {ask_amount}")
        
        if position < 0:
            conv_num = min(conv_limit,- position)
        else:
            conv_num = 0

        logger.print(f"MW price {midprice}, Implied bid {implied_bid}, Implied ask (fair) {implied_ask}, Lowest ask {ask_lowest}")
        
        # Determine the maximum bid and minimum ask after we executed all profitable trades
        # bid_max = max(buy_lst)
        # ask_min = min(sell_lst)

        # Place competitive bids
        # if bid_max < fairprice - 1 and pos_limit - position - buy_quantity > 0:
        #     bid_price = bid_max + 1
        #     bid_amount = pos_limit - position - buy_quantity
        #     orders.append(Order(product,bid_price,bid_amount))
        
        # # Place competitive asks
        # if ask_min > fairprice + 1 and pos_limit + position - sell_quantity > 0:
        #     ask_price = ask_min - 1
        #     ask_amount = pos_limit + position - sell_quantity
        #     orders.append(Order(product,ask_price,-ask_amount))
        
        return orders, conv_num
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))
        
        result = {}
        traderObject = {}

        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        mac_orders, conversions = self.mac_ord(state)
        result["MAGNIFICENT_MACARONS"] = mac_orders
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(traderObject)
        
        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData