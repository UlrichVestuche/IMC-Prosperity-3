from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import math
import jsonpickle

class Trader:

    def ink_mprice(self, state: TradingState):
        product = "SQUID_INK"

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

        return round(2 * mid_price)/2
    
    def ink_window(self, state: TradingState):
        product = "SQUID_INK"

        # window lengths
        len_long = 50
        len_short = 10

        # distance between window curves
        spread_win = 2

        mid_price = self.ink_mprice(state)

        # Decode the dictionary with long and short windows
        if state.traderData:
            dct = jsonpickle.decode(state.traderData)
        else:
            dct = {'long': [], 'short': [], 'trend': 0}

        lst_long = dct['long']
        lst_short = dct['short']

        if len(lst_long) < len_long:
            lst_long.append(mid_price)
        else:
            lst_long.pop(0)
            lst_long.append(mid_price)

        if len(lst_short) < len_short:
            lst_short.append(mid_price)
        else:
            lst_short.pop(0)
            lst_short.append(mid_price)

        if np.mean(lst_short) < np.mean(lst_long) - spread_win:
            new_trend_val = -1
        elif np.mean(lst_short) > np.mean(lst_long) + spread_win:
            new_trend_val = 1
        else:
            new_trend_val = 0

        
        return {'long': lst_long, 'short':lst_short, 'trend': new_trend_val}

    def ink_ord(self, state: TradingState, dict_ink) -> List[Order]:
        product = "SQUID_INK"
        pos_limit = 50
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        # Define the weighted mid price
        mid_price = self.ink_mprice(state)

        # Get the trend value from the dictionary
        trend_val = dict_ink['trend']

        # Find fair price
        fairprice = mid_price
        if trend_val < 0:
            fairprice = mid_price - 0.5
        elif trend_val > 0:
            fairprice = mid_price + 0.5

        # always positive quantities indicating changes in the current position
        buy_quantity = 0
        sell_quantity = 0

        # Define prices just above and just below the fair price
        if fairprice.is_integer():
            above_fprice = int(fairprice)
            below_fprice = int(fairprice-1)
        else:
            above_fprice = math.ceil(fairprice)
            below_fprice = math.floor(fairprice)

        bid_amount = pos_limit - position - buy_quantity
        if bid_amount > 0:
            orders.append(Order(product,below_fprice,bid_amount))

        ask_amount = pos_limit + position - sell_quantity
        if ask_amount > 0:
            orders.append(Order(product,above_fprice,-ask_amount))

        return orders
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))
        dict_ink=self.ink_window(state)

        result = {}

        #result["RAINFOREST_RESIN"] = self.resin_ord(state)
        #result["KELP"] = self.kelp_ord(state, window_lst)
        result["SQUID_INK"] = self.ink_ord(state,dict_ink)
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(dict_ink)
        
		# Sample conversion request.
        conversions = 1

        return result, conversions, traderData