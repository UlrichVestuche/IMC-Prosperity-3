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
    
    def iv_vouchers(self, state: TradingState):
        products = ["VOLCANIC_ROCK_VOUCHER_9750","VOLCANIC_ROCK_VOUCHER_9500","VOLCANIC_ROCK_VOUCHER_10250","VOLCANIC_ROCK_VOUCHER_10000","VOLCANIC_ROCK_VOUCHER_10500"]
        K = {"VOLCANIC_ROCK_VOUCHER_9500": 9500,"VOLCANIC_ROCK_VOUCHER_9750": 9750,"VOLCANIC_ROCK_VOUCHER_10250": 10250,"VOLCANIC_ROCK_VOUCHER_10000": 10000,"VOLCANIC_ROCK_VOUCHER_10500": 10500}
        stock_price = self.r3_mprice["VOLCANIC_ROCK"]
        call_prices = {}

        for prod in products:
            call_prices[prod] = self.r3_mprice[prod]

        low_vol = 10**-7
        high_vol = 10**-4
        max_iterations=200
        tolerance=1e-7
        normal_dist = NormalDist(mu=0, sigma=1)
        T = 7 * 10000
        time_exp = T - 10000 * current_day - (self.state.timestamp)/100
        iv_dct = {}

        for prod in products:
            iv = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
            for _ in range(max_iterations):
                d1 = np.log(stock_price/K[prod])/(np.sqrt(time_exp) * iv) + iv * np.sqrt(time_exp)/2
                d2 = d1 - iv * np.sqrt(time_exp)
                estimated_price = stock_price * normal_dist.cdf(d1) - K[prod] * normal_dist.cdf(d2)
                diff = estimated_price - call_prices[prod]
                if abs(diff) < tolerance:
                    break
                elif diff > 0:
                    high_vol = iv
                else:
                    low_vol = iv
                iv = (low_vol + high_vol) / 2.0
            iv_dct[prod] = iv

        return iv_dct

    def pb_spread(self, state: TradingState, dct):
        # window lengths
        global len_long1
        global len_long2
        
        alpha1 = 1/ len_long1
        alpha2 = 1/ len_long2

        pb1_mprice = self.r2_mprice(state)["PICNIC_BASKET1"]
        pb2_mprice = self.r2_mprice(state)["PICNIC_BASKET2"]
        cro_mprice = self.r2_mprice(state)["CROISSANTS"]
        jam_mprice = self.r2_mprice(state)["JAMS"]
        djem_mprice = self.r2_mprice(state)["DJEMBES"]

        synth_mprice1 = 6 * cro_mprice + 3 * jam_mprice + djem_mprice
        spread_price1 = pb1_mprice - synth_mprice1

        synth_mprice2 = 4 * cro_mprice + 2 * jam_mprice
        spread_price2 = pb2_mprice - synth_mprice2

        win_average1 = dct['average1']
        win_average2 = dct['average2']
        win_average1 = (1 - alpha1) * win_average1 + alpha1 * spread_price1
        win_average2 = (1 - alpha2) * win_average2 + alpha2 * spread_price2

        if spread_price1 - win_average1 > 0:
            z_val1 = 30
        elif spread_price1 - win_average1 < 0:
            z_val1 = -30
        else:
            z_val1 = 0

        if spread_price2 - win_average2 > 0:
            z_val2 = 30
        elif spread_price2 - win_average2 < 0:
            z_val2 = -30
        else:
            z_val2 = 0

        dct['average1'] = win_average1
        dct['stdev1'] = 0
        dct['zscore1'] = z_val1
        dct['average2'] = win_average2
        dct['stdev2'] = 0
        dct['zscore2'] = z_val2

        return dct
    
    def pb_ord(self, state: TradingState, traderObject):
        global price_spread
        
        products = ["PICNIC_BASKET1","PICNIC_BASKET2","CROISSANTS","JAMS","DJEMBES"]
        pos_limit = {}
        pos_limit['PICNIC_BASKET1'] = 60
        pos_limit['PICNIC_BASKET2'] = 100
        pos_limit['CROISSANTS'] = 250
        pos_limit['JAMS'] = 350
        pos_limit['DJEMBES'] = 60

        orders_pb1 = []
        orders_pb2 = []
        orders_cro = []
        orders_jam = []
        orders_djem = []

        # Get current position for product, defaulting to 0 if not present
        pos = {}
        pos['PICNIC_BASKET1'] = state.position["PICNIC_BASKET1"] if "PICNIC_BASKET1" in state.position else 0
        pos['PICNIC_BASKET2'] = state.position["PICNIC_BASKET2"] if "PICNIC_BASKET2" in state.position else 0
        pos['CROISSANTS'] = state.position["CROISSANTS"] if "CROISSANTS" in state.position else 0
        pos['JAMS'] = state.position["JAMS"] if "JAMS" in state.position else 0
        pos['DJEMBES'] = state.position["DJEMBES"] if "DJEMBES" in state.position else 0

        # Decode the dictionary with the spread
        if traderObject.get("zscore1", None) is None:
            traderObject['zscore1'] = 0
            traderObject['average1'] = 0
            traderObject['stdev1'] = 0
            traderObject['zscore2'] = 0
            traderObject['average2'] = 0
            traderObject['stdev2'] = 0

        # Update the dictionary with the current state
        self.pb_spread(state, traderObject)

        # Define mid prices for all products
        pb1_mprice = self.r2_mprice(state)["PICNIC_BASKET1"]
        pb2_mprice = self.r2_mprice(state)["PICNIC_BASKET2"]
        cro_mprice = self.r2_mprice(state)["CROISSANTS"]
        jam_mprice = self.r2_mprice(state)["JAMS"]
        djem_mprice = self.r2_mprice(state)["DJEMBES"]

        # Define fair prices
        fprice = {'PICNIC_BASKET1': pb1_mprice,'PICNIC_BASKET2': pb2_mprice,'CROISSANTS': cro_mprice,'JAMS': jam_mprice,'DJEMBES': djem_mprice}

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
        ratios1 = {'PICNIC_BASKET1': 1,'CROISSANTS': 6,'JAMS': 3,'DJEMBES': 1}
        ratios2 = {'PICNIC_BASKET2': 1,'CROISSANTS': 4,'JAMS': 2}

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
                if p >= fprice[prod] - price_spread[prod]:
                    sell_available[prod] += buy_ord[p]
                    ask_prices.append(p)

            if ask_prices:
                ask_price[prod] = ask_prices[-1]
            
            for p in sell_ord:
                if p <= fprice[prod] + price_spread[prod]:
                    buy_available[prod] -= sell_ord[p]
                    buy_prices.append(p)
            
            if buy_prices:
                bid_price[prod] = buy_prices[-1]

        # Initializing buy and sell quantities
        buy_quantities = {"PICNIC_BASKET1": 0,"PICNIC_BASKET2": 0,"CROISSANTS": 0,"JAMS": 0,"DJEMBES": 0}
        sell_quantities = {"PICNIC_BASKET1": 0,"PICNIC_BASKET2": 0,"CROISSANTS": 0,"JAMS": 0,"DJEMBES": 0}

        for prod in ratios1:
            bid_amount[prod] = min(pos_limit[prod] - pos[prod], buy_available[prod])
            ask_amount[prod] = min(pos_limit[prod] + pos[prod], sell_available[prod])

            bid_factor[prod] = int(bid_amount[prod] / ratios1[prod])
            ask_factor[prod] = int(ask_amount[prod] / ratios1[prod])

        bid_factor_pb1 = min(bid_factor['PICNIC_BASKET1'], ask_factor['CROISSANTS'], ask_factor['JAMS'], ask_factor['DJEMBES'])
        ask_factor_pb1 = min(ask_factor['PICNIC_BASKET1'], bid_factor['CROISSANTS'], bid_factor['JAMS'], bid_factor['DJEMBES'])

        if z_val1 > z_max1 and ask_factor_pb1 > 0: #and pos['PICNIC_BASKET1'] > - pb1_cutoff:
           orders_pb1.append(Order("PICNIC_BASKET1", ask_price['PICNIC_BASKET1'], (-1) * ask_factor_pb1))
           orders_cro.append(Order("CROISSANTS", bid_price['CROISSANTS'], 6 * ask_factor_pb1))
           orders_jam.append(Order("JAMS", bid_price['JAMS'], 3 * ask_factor_pb1))
           orders_djem.append(Order("DJEMBES", bid_price['DJEMBES'], 1 * ask_factor_pb1))
        elif z_val1 < -z_max1 and bid_factor_pb1 > 0: #and pos['PICNIC_BASKET1'] < pb1_cutoff:
           orders_pb1.append(Order("PICNIC_BASKET1", bid_price['PICNIC_BASKET1'], 1 * bid_factor_pb1))
           orders_cro.append(Order("CROISSANTS", ask_price['CROISSANTS'], (-6) * bid_factor_pb1))
           orders_jam.append(Order("JAMS", ask_price['JAMS'], (-3) * bid_factor_pb1))
           orders_djem.append(Order("DJEMBES", ask_price['DJEMBES'], (-1) * bid_factor_pb1))

        buy_quantities = {"PICNIC_BASKET1": bid_factor_pb1,"PICNIC_BASKET2": 0,"CROISSANTS": 6 * ask_factor_pb1,"JAMS": 3 * ask_factor_pb1,"DJEMBES": ask_factor_pb1}
        sell_quantities = {"PICNIC_BASKET1": ask_factor_pb1,"PICNIC_BASKET2": 0,"CROISSANTS": 6 * bid_factor_pb1,"JAMS": 3 * bid_factor_pb1,"DJEMBES": bid_factor_pb1}

        for prod in ratios2:
            bid_amount2[prod] = min(pos_limit[prod] - pos[prod] - buy_quantities[prod], buy_available[prod] - buy_quantities[prod])
            ask_amount2[prod] = min(pos_limit[prod] + pos[prod] - sell_quantities[prod], sell_available[prod] - sell_quantities[prod])

            bid_factor2[prod] = int(bid_amount2[prod] / ratios2[prod])
            ask_factor2[prod] = int(ask_amount2[prod] / ratios2[prod])

        bid_factor_pb2 = min(bid_factor2['PICNIC_BASKET2'], ask_factor2['CROISSANTS'], ask_factor2['JAMS'])
        ask_factor_pb2 = min(ask_factor2['PICNIC_BASKET2'], bid_factor2['CROISSANTS'], bid_factor2['JAMS'])
        
        if z_val2 > z_max2 and ask_factor_pb2 > 0: #and pos['PICNIC_BASKET2'] > - pb2_cutoff:
            orders_pb2.append(Order("PICNIC_BASKET2", ask_price['PICNIC_BASKET2'], (-1) * ask_factor_pb2))
            orders_cro.append(Order("CROISSANTS", bid_price['CROISSANTS'], 4 * ask_factor_pb2))
            orders_jam.append(Order("JAMS", bid_price['JAMS'], 2 * ask_factor_pb2))
        elif z_val2 < -z_max2 and bid_factor_pb2 > 0: #and pos['PICNIC_BASKET2'] < pb2_cutoff:
            orders_pb2.append(Order("PICNIC_BASKET2", bid_price['PICNIC_BASKET2'], 1 * bid_factor_pb2))
            orders_cro.append(Order("CROISSANTS", ask_price['CROISSANTS'], (-4) * bid_factor_pb2))
            orders_jam.append(Order("JAMS", ask_price['JAMS'], (-2) * bid_factor_pb2))

        return {'pb1': orders_pb1, 'pb2': orders_pb2, 'cro': orders_cro,'jam': orders_jam, 'djem': orders_djem}
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))

        result = {}
        traderObject = {}
        conversions = 1
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        PB_dict = self.pb_ord(state, traderObject)

        #result["RAINFOREST_RESIN"] = self.resin_ord(state)
        #result["KELP"] = self.kelp_ord(state, window_lst)
        result["PICNIC_BASKET1"] = PB_dict['pb1']
        result["PICNIC_BASKET2"] = PB_dict['pb2']
        result["CROISSANTS"] = PB_dict['cro']
        result["JAMS"] = PB_dict['jam']
        result["DJEMBES"] = PB_dict['djem']

        logger.print(PB_dict['pb2'])
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(traderObject)
    

        logger.flush(state, result, conversions, traderData)


        return result, conversions, traderData