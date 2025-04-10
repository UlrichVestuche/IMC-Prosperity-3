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
    
    def ink_window(self, state: TradingState, dct):
        product = "SQUID_INK"

        # window lengths
        len_long = 10
        len_short = 10

        mid_price = self.ink_mprice(state)

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

        mean_long = sum(lst_long) / len(lst_long)
        stdev_sq_long = 0

        for p in lst_long:
            stdev_sq_long += (p - mean_long) ** 2

        #stdev_long = math.sqrt(stdev_sq_long/len(lst_long))
        stdev_long = 1
    
        if stdev_long > 0:
            z_val = (mid_price - mean_long) / stdev_long
        else: 
            z_val = 0

        dct['long'] = lst_long
        dct['short'] = lst_short
        dct['zscore'] = z_val

        return dct

    def ink_ord(self, state: TradingState):
        product = "SQUID_INK"
        pos_limit = 50
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        # Decode the dictionary with long and short windows
        if state.traderData:
            dict_ink_prev = jsonpickle.decode(state.traderData)
        else:
            dict_ink_prev = {'long': [], 'short': [],'zscore': 0, 'index': 0}

        # Update the dictionary with the current state
        dict_ink = self.ink_window(state, dict_ink_prev)

        # Define the weighted mid price
        mid_price = self.ink_mprice(state)

        # Find fair price
        fairprice = mid_price

        #Import the long and short windows
        z_val = dict_ink['zscore']
        ind_val = dict_ink['index']

        bid_amount = pos_limit - position
        ask_amount = pos_limit + position

        # Define prices just above and just below the fair price
        if fairprice.is_integer():
            above_fprice = int(fairprice + 1)
            below_fprice = int(fairprice - 1)
        else:
            above_fprice = math.ceil(fairprice)
            below_fprice = math.floor(fairprice)
        

        if ind_val == 0 and z_val > 2:
            if ask_amount > 0:
                orders.append(Order(product, above_fprice, -ask_amount))
            ind_val = -1
        elif ind_val == -1 and z_val < 0.5:
            if bid_amount > 0:
                orders.append(Order(product, below_fprice, bid_amount))
            ind_val = 0
        elif ind_val == 0 and z_val < -2:
            if bid_amount > 0:
                orders.append(Order(product, below_fprice, bid_amount))
            ind_val = 1
        elif ind_val == 1 and z_val > -0.5:
            if ask_amount > 0:
                orders.append(Order(product, above_fprice, -ask_amount))
            ind_val = 0
        #else:
        #    if bid_amount > 0:
        #        orders.append(Order(product,below_fprice,bid_amount))
        #    if ask_amount > 0:
        #        orders.append(Order(product,above_fprice,-ask_amount))

        dict_ink['index'] = ind_val
        
        # always positive quantities indicating changes in the current position
        #buy_quantity = 0
        #sell_quantity = 0

        # Define prices just above and just below the fair price
        #if fairprice.is_integer():
        #    above_fprice = int(fairprice + 1)
        #    below_fprice = int(fairprice - 1)
        #else:
        #    above_fprice = math.ceil(fairprice)
        #    below_fprice = math.floor(fairprice)

        #bid_amount = pos_limit - position - buy_quantity
        #if bid_amount > 0:
        #    orders.append(Order(product,below_fprice,bid_amount))

        #ask_amount = pos_limit + position - sell_quantity
        #if ask_amount > 0:
        #    orders.append(Order(product,above_fprice,-ask_amount))

        return {'Orders': orders, 'Dict_Win': dict_ink}
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))

        result = {}

        Ink_dict = self.ink_ord(state)

        #result["RAINFOREST_RESIN"] = self.resin_ord(state)
        #result["KELP"] = self.kelp_ord(state, window_lst)
        result["SQUID_INK"] = Ink_dict['Orders']

        #print(Ink_dict['Orders'])
    
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(Ink_dict['Dict_Win'])
        
        print(Ink_dict['Dict_Win'])

		# Sample conversion request.
        conversions = 1

        return result, conversions, traderData