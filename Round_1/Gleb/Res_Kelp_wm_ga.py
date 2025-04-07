from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import math
import jsonpickle

class Trader:

    def kelp_mprice(self, state: TradingState):
        product = "KELP"

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
    
    def kelp_window(self, state: TradingState):
        product = "KELP"

        # Determine actual mid price
        mid_price = self.kelp_mprice(state)

        # Decode the dictionary with window
        if state.traderData:
            dct = jsonpickle.decode(state.traderData)
        else:
            dct = {'mid': [], 'frame': 0}

        # Extract the old window
        lst_mid_old = dct['mid']

        # Determine the frame number
        frame = dct['frame']
        frame += 1

        # Store actual mid price in the new list
        lst_mid_new = []
        lst_mid_new.append(mid_price)

        mid_price_old_actual = lst_mid_old[0] if len(lst_mid_old) > 0 else 0
        mid_price_old = lst_mid_old[-1] if len(lst_mid_old) > 0 else 0

        if mid_price_old and mid_price_old_actual:
            if mid_price >= mid_price_old + 1.4 and mid_price != mid_price_old_actual:
                mid_price = mid_price_old + 0
        
            if mid_price <= mid_price_old - 1.4 and mid_price != mid_price_old_actual:
                mid_price = mid_price_old - 0
        
        lst_mid_new.append(mid_price)
    
        return {'mid': lst_mid_new, 'frame': frame}

    def kelp_ord(self, state: TradingState, window_lst) -> List[Order]:
        product = "KELP"
        pos_limit = 50
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        # Define the weighted mid price
        mid_price = window_lst[-1]

        # Find fair price
        fairprice = mid_price

        # Define prices just above and just below the fair price
        if fairprice.is_integer():
            above_fprice = int(fairprice + 1)
            below_fprice = int(fairprice - 1)
        else:
            above_fprice = math.ceil(fairprice + 0.5)
            below_fprice = math.floor(fairprice - 0.5)
        
        # always positive quantities indicating changes in the current position
        buy_quantity = 0
        sell_quantity = 0

        bid_amount = pos_limit - position - buy_quantity
        if bid_amount > 0:
            orders.append(Order(product,below_fprice,bid_amount))

        ask_amount = pos_limit + position - sell_quantity
        if ask_amount > 0:
            orders.append(Order(product,above_fprice,-ask_amount))

        return orders



    def resin_ord(self, state: TradingState) -> List[Order]:
        product = "RAINFOREST_RESIN"
        pos_limit = 50
        fairprice = 10000
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        buy_ord = state.order_depths[product].buy_orders
        sell_ord = state.order_depths[product].sell_orders
       
        buy_lst = []
        sell_lst = []
        
        # always positive quantities indicating changes in the current position
        buy_quantity = 0
        sell_quantity = 0

        # If there are bids above fair price, sell the maximum possible volume
        for p in buy_ord:
            if p > fairprice and pos_limit + position - sell_quantity > 0:
                sell_amount = min(buy_ord[p], pos_limit + position - sell_quantity)
                orders.append(Order(product,p,-sell_amount))
                sell_quantity += sell_amount
                if sell_amount < buy_ord[p]:
                    buy_lst.append(p)
            else:
                buy_lst.append(p)
        
        # If there are asks below fair price, buy the maximum possible volume
        for p in sell_ord:
            if p < fairprice and pos_limit - position - buy_quantity > 0:
                buy_amount = min(-sell_ord[p], pos_limit - position - buy_quantity)
                orders.append(Order(product,p,buy_amount))
                buy_quantity += buy_amount
                if buy_amount < -sell_ord[p]:
                    sell_lst.append(p)
            else:
                sell_lst.append(p)

        # If all available positions were closed, put out Orders at maximum profit
        if not buy_lst and pos_limit - position - buy_quantity > 0:
            bid_amount = pos_limit - position - buy_quantity
            orders.append(Order(product,1,bid_amount))

        if not sell_lst and pos_limit + position - sell_quantity > 0:
            ask_amount = pos_limit + position - sell_quantity
            orders.append(Order(product,20000,-ask_amount))
        
        # Determine the maximum bid and minimum ask after we executed all profitable trades
        bid_max = max(buy_lst)
        ask_min = min(sell_lst)

        # Place competitive bids
        if bid_max < fairprice - 1 and pos_limit - position - buy_quantity > 0:
            bid_price = bid_max + 1
            bid_amount = pos_limit - position - buy_quantity
            orders.append(Order(product,bid_price,bid_amount))
        
        # Place competitive asks
        if ask_min > fairprice + 1 and pos_limit + position - sell_quantity > 0:
            ask_price = ask_min - 1
            ask_amount = pos_limit + position - sell_quantity
            orders.append(Order(product,ask_price,-ask_amount))
        
        return orders
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))
        
        result = {}

        dict_kelp = self.kelp_window(state)

        window_lst = dict_kelp['mid']

        result["RAINFOREST_RESIN"] = self.resin_ord(state)
        result["KELP"] = self.kelp_ord(state, window_lst)
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(dict_kelp)
        
		# Sample conversion request.
        conversions = 1

        return result, conversions, traderData