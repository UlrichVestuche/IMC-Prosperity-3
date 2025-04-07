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

        buy_lst = list(buy_ord.keys())
        sell_lst = list(sell_ord.keys())

        #mid_weight_curr = 0
        #total_weight = 0

        mid_price = int((max(buy_lst) + min(sell_lst)) / 2)

        #for p in buy_lst:
        #    if buy_ord[p] > 0:
        #        mid_weight_curr += p * buy_ord[p]
        #        total_weight += buy_ord[p]

        #for p in sell_lst:
        #    if -sell_ord[p] > 0:
        #        mid_weight_curr -= p * sell_ord[p]
        #        total_weight -= sell_ord[p]

        return mid_price

    def kelp_window(self, state: TradingState):
        product = "KELP"

        # window length
        len_mid = 4

        mid_price = self.kelp_mprice(state)

        # Decode the dictionary with window
        if state.traderData:
            dct = jsonpickle.decode(state.traderData)
        else:
            dct = {'mid': []}

        lst_mid = dct['mid']
        
        if len(lst_mid) < len_mid:
            lst_mid.append(mid_price)
        else:
            lst_mid.pop(0)
            lst_mid.append(mid_price)
        
        return {'mid': lst_mid}

    def kelp_ord(self, state: TradingState) -> List[Order]:
        product = "KELP"
        pos_limit = 50
        orders = []
        
        # Buy and sell limits
        buysell_level = 20
        buysell_max = 15
        stdev = 0


        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        buy_ord = state.order_depths[product].buy_orders
        sell_ord = state.order_depths[product].sell_orders

        buy_lst = []
        sell_lst = []

        # Define the list with the last four mid prices
        window_lst = self.kelp_window(state)['mid']

        # Define linear regression parameters
        beta_0 = 17.63806394782273
        beta_lst = [0.99127796]

        # Find fair price
        fairprice = beta_0 + beta_lst[0] * np.mean(window_lst)

        # Define the price just above the fair price
        above_fprice = math.ceil(fairprice + 1)

        # Define the price just below the fair price
        below_fprice = math.floor(fairprice)
        
        # always positive quantities indicating changes in the current position
        buy_quantity = 0
        sell_quantity = 0

        # Buy or sell close to fairprice if the position is too close to the limits
        if 0 < position: #<= dump_Level_mid:
            sell_amount = position
            orders.append(Order(product, above_fprice,-sell_amount))
            sell_quantity += sell_amount


        if 0 > position: # >= -dump_Level_mid:
            buy_amount = -position
            orders.append(Order(product, below_fprice,buy_amount))
            buy_quantity += buy_amount

        # If there are asks below fair price, buy the maximum possible volume
        for p in sell_ord:
            if p < fairprice - stdev and pos_limit - position - buy_quantity > 0 and position <= buysell_level:
                buy_amount = min(buysell_max, -sell_ord[p], pos_limit - position - buy_quantity)
                orders.append(Order(product,p,buy_amount))
                buy_quantity += buy_amount
                if buy_amount < -sell_ord[p]:
                    sell_lst.append(p)
                    break
            else:
                sell_lst.append(p)

        # If there are bids above fair price, sell the maximum possible volume
        for p in buy_ord:
            if p > fairprice + stdev and pos_limit + position - sell_quantity > 0 and position >= buysell_level:
                sell_amount = min(buysell_max, buy_ord[p], pos_limit + position - sell_quantity)
                orders.append(Order(product,p,-sell_amount))
                sell_quantity += sell_amount
                if sell_amount < buy_ord[p]:
                    buy_lst.append(p)
                    break
            else:
                buy_lst.append(p)

        if buy_lst:
            # Determine the maximum bid after we executed all profitable trades
            bid_max = max(buy_lst)

            # Place competitive bids
            if bid_max + 1 < fairprice - stdev  and pos_limit - position - buy_quantity > 0:
                bid_price = bid_max + 1
                bid_amount = min(buysell_max, pos_limit - position - buy_quantity)
                orders.append(Order(product,bid_price,bid_amount))
        if sell_lst:
            # Determine the minimum ask after we executed all profitable trades
            ask_min = min(sell_lst)
            
            # Place competitive asks
            if ask_min - 1 > fairprice + stdev and pos_limit + position - sell_quantity > 0:
                ask_price = ask_min - 1
                ask_amount = min(buysell_max, pos_limit + position - sell_quantity)
                orders.append(Order(product,ask_price,-ask_amount))

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
        if bid_max + 1 < fairprice and pos_limit - position - buy_quantity > 0:
            bid_price = bid_max + 1
            bid_amount = pos_limit - position - buy_quantity
            orders.append(Order(product,bid_price,bid_amount))
        
        # Place competitive asks
        if ask_min - 1 > fairprice and pos_limit + position - sell_quantity > 0:
            ask_price = ask_min - 1
            ask_amount = pos_limit + position - sell_quantity
            orders.append(Order(product,ask_price,-ask_amount))
        
        return orders
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))
        
        result = {}

        result["RAINFOREST_RESIN"] = self.resin_ord(state)    
        result["KELP"] = self.kelp_ord(state) 
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(self.kelp_window(state))
        
		# Sample conversion request.
        conversions = 1

        return result, conversions, traderData