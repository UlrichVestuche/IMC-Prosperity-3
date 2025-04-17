from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import math
import jsonpickle

import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from statistics import NormalDist

# Global variables
current_day = 0

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

    def r3_mprice(self, state: TradingState):
        #Compute the weighted mid price for all new products in Round 2

        products = ["VOLCANIC_ROCK","VOLCANIC_ROCK_VOUCHER_9750","VOLCANIC_ROCK_VOUCHER_9500","VOLCANIC_ROCK_VOUCHER_10250","VOLCANIC_ROCK_VOUCHER_10000","VOLCANIC_ROCK_VOUCHER_10500"]
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
    

    def rock_spread(self, state: TradingState, dct):
        # window lengths
        len_long1 = 150
        len_long2 = 150
        
        alpha1 = 1/ len_long1
        alpha2 = 1/ len_long2

        ratios1 = {"VOLCANIC_ROCK": 1,"VOLCANIC_ROCK_VOUCHER_10500": 2}
        ratios2 = {"VOLCANIC_ROCK": 1,"VOLCANIC_ROCK_VOUCHER_10250": 2}
        intercept1 = 10300
        intercept2 = 9983
        band1 = 4
        band2 = 5

        rock_mprice = self.r3_mprice(state)["VOLCANIC_ROCK"]
        v102_mprice = self.r3_mprice(state)["VOLCANIC_ROCK_VOUCHER_10250"]
        v105_mprice = self.r3_mprice(state)["VOLCANIC_ROCK_VOUCHER_10500"]

        spread_price1 = ratios1["VOLCANIC_ROCK"] * rock_mprice - ratios1["VOLCANIC_ROCK_VOUCHER_10500"] * v105_mprice

        spread_price2 = ratios2["VOLCANIC_ROCK"] * rock_mprice - ratios2["VOLCANIC_ROCK_VOUCHER_10250"] * v102_mprice

        win_average1 = dct['average1']
        win_stdev1 = dct['stdev1']
        win_average2 = dct['average2']
        win_stdev2 = dct['stdev2']

        win_average1 = (1 - alpha1) * win_average1 + alpha1 * spread_price1
        win_stdev1 = math.sqrt((1 - alpha1) * win_stdev1**2 + alpha1 * (spread_price1 - win_average1)**2)

        win_average2 = (1 - alpha2) * win_average2 + alpha2 * spread_price2
        win_stdev2 = math.sqrt((1 - alpha2) * win_stdev2**2 + alpha2 * (spread_price2 - win_average2)**2)

        if win_stdev1:
            z_val1 = (spread_price1 - win_average1) / win_stdev1
        else:
            z_val1 = 0

        if win_stdev2:
            z_val2 = (spread_price2 - win_average2) / win_stdev2
        else:
            z_val2 = 0

        dct['average1'] = win_average1
        dct['stdev1'] = win_stdev1
        dct['zscore1'] = z_val1
        dct['average2'] = win_average2
        dct['stdev2'] = win_stdev2
        dct['zscore2'] = z_val2

        return dct
    
    def rock_ord(self, state: TradingState, traderObject):
        global price_spread
        z_max1 = 1.8
        z_max2 = 1.8
        
        products = ["VOLCANIC_ROCK","VOLCANIC_ROCK_VOUCHER_9750","VOLCANIC_ROCK_VOUCHER_9500","VOLCANIC_ROCK_VOUCHER_10250","VOLCANIC_ROCK_VOUCHER_10000","VOLCANIC_ROCK_VOUCHER_10500"]
        pos_limit = {}
        pos_limit["VOLCANIC_ROCK"] = 400
        pos_limit["VOLCANIC_ROCK_VOUCHER_9500"] = 200
        pos_limit["VOLCANIC_ROCK_VOUCHER_9750"] = 200
        pos_limit["VOLCANIC_ROCK_VOUCHER_10000"] = 200
        pos_limit["VOLCANIC_ROCK_VOUCHER_10250"] = 200
        pos_limit["VOLCANIC_ROCK_VOUCHER_10500"] = 200

        orders_rock = []
        orders_v102 = []
        orders_v105 = []

        # Get current position for product, defaulting to 0 if not present
        pos = {}
        for prod in products:
            pos[prod] = state.position[prod] if prod in state.position else 0

        # Decode the dictionary with the spread
        if traderObject.get("zscore1", None) is None:
            traderObject['zscore1'] = 0
            traderObject['average1'] = 0
            traderObject['stdev1'] = 0
            traderObject['zscore2'] = 0
            traderObject['average2'] = 0
            traderObject['stdev2'] = 0

        # Update the dictionary with the current state
        self.rock_spread(state, traderObject)

        # Define fair prices
        fprice = {}
        for prod in products:
          fprice[prod] = self.r3_mprice(state)[prod]
        
        # Import z values
        z_val1 = traderObject['zscore1']
        z_val2 = traderObject['zscore2']

        # Define the bid factors
        bid_amount = {}
        ask_amount = {}
        bid_amount2 = {}
        ask_amount2 = {}
        bid_factor = {}
        ask_factor = {}
        bid_factor2 = {}
        ask_factor2 = {}
        ratios1 = {"VOLCANIC_ROCK": 1,"VOLCANIC_ROCK_VOUCHER_10500": 2}
        ratios2 = {"VOLCANIC_ROCK": 1,"VOLCANIC_ROCK_VOUCHER_10250": 2}

        # Determine the bid price depending on the needed quantity
        ask_price = {}
        bid_price = {}
        sell_available = {}
        buy_available = {}

        for prod in products:
            sell_available[prod] = 0
            buy_available[prod] = 0

            buy_ord = state.order_depths[prod].buy_orders
            sell_ord = state.order_depths[prod].sell_orders
            
            ask_prices = []
            buy_prices = []
            
            for p in buy_ord:    
                if p >= fprice[prod] - 1:
                    sell_available[prod] += buy_ord[p]
                    ask_prices.append(p)

            if ask_prices:
                ask_price[prod] = ask_prices[-1]
            
            for p in sell_ord:
                if p <= fprice[prod] + 1:
                    buy_available[prod] -= sell_ord[p]
                    buy_prices.append(p)
            
            if buy_prices:
                bid_price[prod] = buy_prices[-1]

        # Initializing buy and sell quantities
        buy_quantities = {"VOLCANIC_ROCK": 0,"VOLCANIC_ROCK_VOUCHER_10250": 0,"VOLCANIC_ROCK_VOUCHER_10500": 0}
        sell_quantities = {"VOLCANIC_ROCK": 0,"VOLCANIC_ROCK_VOUCHER_10250": 0,"VOLCANIC_ROCK_VOUCHER_10500": 0}

        for prod in ratios1:
            bid_amount[prod] = min(pos_limit[prod] - pos[prod], buy_available[prod])
            ask_amount[prod] = min(pos_limit[prod] + pos[prod], sell_available[prod])

            bid_factor[prod] = int(bid_amount[prod] / ratios1[prod])
            ask_factor[prod] = int(ask_amount[prod] / ratios1[prod])

        bid_factor_r1 = min(bid_factor["VOLCANIC_ROCK"], ask_factor["VOLCANIC_ROCK_VOUCHER_10500"])
        ask_factor_r1 = min(ask_factor["VOLCANIC_ROCK"], bid_factor["VOLCANIC_ROCK_VOUCHER_10500"])

        if z_val1 > z_max1 and ask_factor_r1 > 0:
           orders_rock.append(Order("VOLCANIC_ROCK", ask_price["VOLCANIC_ROCK"], (-ratios1["VOLCANIC_ROCK"]) * ask_factor_r1))
           orders_v105.append(Order("VOLCANIC_ROCK_VOUCHER_10500", bid_price["VOLCANIC_ROCK_VOUCHER_10500"], ratios1["VOLCANIC_ROCK_VOUCHER_10500"] * ask_factor_r1))
        elif z_val1 < -z_max1 and bid_factor_r1 > 0:
           orders_rock.append(Order("VOLCANIC_ROCK", bid_price["VOLCANIC_ROCK"], ratios1["VOLCANIC_ROCK"] * bid_factor_r1))
           orders_v105.append(Order("VOLCANIC_ROCK_VOUCHER_10500", ask_price["VOLCANIC_ROCK_VOUCHER_10500"], (-ratios1["VOLCANIC_ROCK_VOUCHER_10500"]) * bid_factor_r1))

        buy_quantities = {"VOLCANIC_ROCK": ratios1["VOLCANIC_ROCK"] * bid_factor_r1,"VOLCANIC_ROCK_VOUCHER_10500": ratios1["VOLCANIC_ROCK_VOUCHER_10500"] * ask_factor_r1, "VOLCANIC_ROCK_VOUCHER_10250": 0}
        sell_quantities = {"VOLCANIC_ROCK": ratios1["VOLCANIC_ROCK"] * ask_factor_r1,"VOLCANIC_ROCK_VOUCHER_10500": ratios1["VOLCANIC_ROCK_VOUCHER_10500"] * bid_factor_r1, "VOLCANIC_ROCK_VOUCHER_10250": 0}

        for prod in ratios2:
            bid_amount2[prod] = min(pos_limit[prod] - pos[prod] - buy_quantities[prod], buy_available[prod] - buy_quantities[prod])
            ask_amount2[prod] = min(pos_limit[prod] + pos[prod] - sell_quantities[prod], sell_available[prod] - sell_quantities[prod])

            bid_factor2[prod] = int(bid_amount2[prod] / ratios2[prod])
            ask_factor2[prod] = int(ask_amount2[prod] / ratios2[prod])

        bid_factor_r2 = min(bid_factor2["VOLCANIC_ROCK"], ask_factor2["VOLCANIC_ROCK_VOUCHER_10250"])
        ask_factor_r2 = min(ask_factor2["VOLCANIC_ROCK"], bid_factor2["VOLCANIC_ROCK_VOUCHER_10250"])
        
        if z_val2 > z_max2 and ask_factor_r2 > 0:
            orders_rock.append(Order("VOLCANIC_ROCK", ask_price["VOLCANIC_ROCK"], -ratios2["VOLCANIC_ROCK"] * ask_factor_r2))
            orders_v102.append(Order("VOLCANIC_ROCK_VOUCHER_10250", bid_price["VOLCANIC_ROCK_VOUCHER_10250"], ratios2["VOLCANIC_ROCK_VOUCHER_10250"] * ask_factor_r2))
        elif z_val2 < -z_max2 and bid_factor_r2 > 0:
             orders_rock.append(Order("VOLCANIC_ROCK", bid_price["VOLCANIC_ROCK"], ratios2["VOLCANIC_ROCK"] * bid_factor_r2))
             orders_v102.append(Order("VOLCANIC_ROCK_VOUCHER_10250", ask_price["VOLCANIC_ROCK_VOUCHER_10250"], -ratios2["VOLCANIC_ROCK_VOUCHER_10250"] * bid_factor_r2))

        return {'rock': orders_rock, 'v102': orders_v102, 'v105': orders_v105}
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))

        result = {}
        traderObject = {}
        conversions = 1
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        PB_dict = self.rock_ord(state, traderObject)

        result["VOLCANIC_ROCK"] = PB_dict['rock']
        result["VOLCANIC_ROCK_VOUCHER_10500"] = PB_dict['v105']
        result["VOLCANIC_ROCK_VOUCHER_10250"] = PB_dict['v102']

        logger.print(PB_dict['rock'])
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(traderObject)
    

        logger.flush(state, result, conversions, traderData)


        return result, conversions, traderData