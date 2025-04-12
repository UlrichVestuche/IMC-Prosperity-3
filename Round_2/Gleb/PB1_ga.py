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

    def r2_mprice(self, state: TradingState):
        #Compute the weighted mid price for all new products in Round 2

        products = ["PICNIC_BASKET1","PICNIC_BASKET2","CROISSANTS","JAMS","DJEMBES"]
        mid_price = {}

        for product in products:
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

            mid_price[product] = mid_weight_curr / total_weight if total_weight > 0 else 0

        return mid_price
    
    def pb1_spread(self, state: TradingState, dct):
        #product = "PICNIC_BASKET1"

        # window lengths
        len_long = 100
        len_short = 1

        pb1_mprice = self.r2_mprice(state)["PICNIC_BASKET1"]
        cro_mprice = self.r2_mprice(state)["CROISSANTS"]
        jam_mprice = self.r2_mprice(state)["JAMS"]
        djem_mprice = self.r2_mprice(state)["DJEMBES"]

        synth_mprice = 6 * cro_mprice + 3 * jam_mprice + djem_mprice
        spread_price = pb1_mprice - synth_mprice


        lst_long = dct['long']
        lst_short = dct['short']

        if len(lst_long) < len_long:
            lst_long.append(spread_price)
        else:
            lst_long.pop(0)
            lst_long.append(spread_price)

        if len(lst_short) < len_short:
            lst_short.append(spread_price)
        else:
            lst_short.pop(0)
            lst_short.append(spread_price)

        mean_long = sum(lst_long) / len(lst_long)
        stdev_sq_long = 0

        for p in lst_long:
            stdev_sq_long += (p - mean_long) ** 2

        stdev_long = math.sqrt(stdev_sq_long/len(lst_long))
        #stdev_long = 1
    
        if stdev_long > 0:
            z_val = (spread_price - mean_long) / stdev_long
        else: 
            z_val = 0

        dct['long'] = lst_long
        dct['short'] = lst_short
        dct['zscore'] = z_val

        return dct

    def pb1_ord(self, state: TradingState):
        #product = "PICNIC_BASKET1"
        pos_limit = {}
        pos_limit['PICNIC_BASKET1'] = 60
        pos_limit['CROISSANTS'] = 250
        pos_limit['JAMS'] = 350
        pos_limit['DJEMBES'] = 60

        orders_pb1 = []
        orders_cro = []
        orders_jam = []
        orders_djem = []

        # Get current position for product, defaulting to 0 if not present
        pos = {}
        pos['PICNIC_BASKET1'] = state.position["PICNIC_BASKET1"] if "PICNIC_BASKET1" in state.position else 0
        pos['CROISSANTS'] = state.position["CROISSANTS"] if "CROISSANTS" in state.position else 0
        pos['JAMS'] = state.position["JAMS"] if "JAMS" in state.position else 0
        pos['DJEMBES'] = state.position["DJEMBES"] if "DJEMBES" in state.position else 0

        # Decode the dictionary with the spread
        if state.traderData:
            dict_spread1_prev = jsonpickle.decode(state.traderData)
        else:
            dict_spread1_prev = {'long': [], 'short': [],'zscore': 0}

        # Update the dictionary with the current state
        dict_spread1 = self.pb1_spread(state, dict_spread1_prev)

        # Define mid prices for all products
        pb1_mprice = self.r2_mprice(state)["PICNIC_BASKET1"]
        cro_mprice = self.r2_mprice(state)["CROISSANTS"]
        jam_mprice = self.r2_mprice(state)["JAMS"]
        djem_mprice = self.r2_mprice(state)["DJEMBES"]

        # Define fair prices
        fprice = {'PICNIC_BASKET1': pb1_mprice,'CROISSANTS': cro_mprice,'JAMS': jam_mprice,'DJEMBES': djem_mprice}

        # Import the z value
        z_val = dict_spread1['zscore']

        # Define the bid factors
        bid_amount = {}
        ask_amount = {}
        bid_factor = {}
        ask_factor = {}
        ratios = {'PICNIC_BASKET1': 1,'CROISSANTS': 6,'JAMS': 3,'DJEMBES': 1}

        # Determine the bid price depending on the needed quantity
        ask_price = {}
        bid_price = {}

        for prod in fprice:
            sell_available = 0
            buy_available = 0

            buy_ord = state.order_depths[prod].buy_orders
            sell_ord = state.order_depths[prod].sell_orders
            
            ask_prices = []
            buy_prices = []

            for p in buy_ord:
                if p >= fprice[prod] - 2:
                    sell_available += buy_ord[p]
                    ask_prices.append(p)

            if ask_prices:
                ask_price[prod] = ask_prices[-1]
            
            for p in sell_ord:
                if p <= fprice[prod] + 2:
                    buy_available -= sell_ord[p]
                    buy_prices.append(p)
            
            if buy_prices:
                bid_price[prod] = buy_prices[-1]

            bid_amount[prod] = min(pos_limit[prod] - pos[prod], buy_available)
            ask_amount[prod] = min(pos_limit[prod] + pos[prod], sell_available)

            bid_factor[prod] = int(bid_amount[prod] / ratios[prod])
            ask_factor[prod] = int(ask_amount[prod] / ratios[prod])

        bid_factor_pb1 = min(bid_factor['PICNIC_BASKET1'], ask_factor['CROISSANTS'], ask_factor['JAMS'], ask_factor['DJEMBES'])
        ask_factor_pb1 = min(ask_factor['PICNIC_BASKET1'], bid_factor['CROISSANTS'], bid_factor['JAMS'], bid_factor['DJEMBES'])

        if z_val > 1.5 and ask_factor_pb1 > 0:
            orders_pb1.append(Order("PICNIC_BASKET1", ask_price['PICNIC_BASKET1'], (-1) * ask_factor_pb1))
            orders_cro.append(Order("CROISSANTS", bid_price['CROISSANTS'], 6 * ask_factor_pb1))
            orders_jam.append(Order("JAMS", bid_price['JAMS'], 3 * ask_factor_pb1))
            orders_djem.append(Order("DJEMBES", bid_price['DJEMBES'], 1 * ask_factor_pb1))
        elif z_val < -1.5 and bid_factor_pb1 > 0:
            orders_pb1.append(Order("PICNIC_BASKET1", bid_price['PICNIC_BASKET1'], 1 * bid_factor_pb1))
            orders_cro.append(Order("CROISSANTS", ask_price['CROISSANTS'], (-6) * bid_factor_pb1))
            orders_jam.append(Order("JAMS", ask_price['JAMS'], (-3) * bid_factor_pb1))
            orders_djem.append(Order("DJEMBES", ask_price['DJEMBES'], (-1) * bid_factor_pb1))
        
        return {'pb1': orders_pb1, 'cro': orders_cro,'jam': orders_jam, 'djem': orders_djem, 'Dict_Spread1': dict_spread1}
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))

        result = {}

        PB1_dict = self.pb1_ord(state)

        #result["RAINFOREST_RESIN"] = self.resin_ord(state)
        #result["KELP"] = self.kelp_ord(state, window_lst)
        result["PICNIC_BASKET1"] = PB1_dict['pb1']
        result["CROISSANTS"] = PB1_dict['cro']
        result["JAMS"] = PB1_dict['jam']
        result["DJEMBES"] = PB1_dict['djem']
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(PB1_dict['Dict_Spread1'])
        

		# Sample conversion request.
        conversions = 1

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData