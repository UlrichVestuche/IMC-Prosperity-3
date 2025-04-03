from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import math

class Trader:

    def kelp_ord(self, state: TradingState) -> List[Order]:
        product = "KELP"
        pos_limit = 50
        dump_range = 5
        dump_amount = 40
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        buy_ord = state.order_depths[product].buy_orders
        sell_ord = state.order_depths[product].sell_orders

        buy_lst = list(buy_ord.keys())
        sell_lst = list(sell_ord.keys())

        b_max = max(buy_lst)
        s_min = min(sell_lst)

        # fairprice = int((b_max * buy_ord[b_max] + s_min * sell_ord[s_min])/(buy_ord[b_max] + sell_ord[s_min]))
        fairprice = int((b_max + s_min)/2)
        
        # always positive quantities indicating changes in the current position
        buy_quantity = 0
        sell_quantity = 0
        
        if buy_ord[b_max] < 1 and pos_limit + position - sell_quantity > 0:
            sell_amount = min(buy_ord[b_max], pos_limit + position - sell_quantity)
            orders.append(Order(product,b_max,-sell_amount))
            sell_quantity += sell_amount
            if sell_amount == buy_ord[b_max]:
                buy_lst.pop(0)
        
        if -sell_ord[s_min] < 1 and pos_limit - position - buy_quantity > 0:
            buy_amount = min(-sell_ord[s_min], pos_limit - position - buy_quantity)
            orders.append(Order(product,s_min,buy_amount))
            buy_quantity += buy_amount
            if buy_amount == -sell_ord[s_min]:
                sell_lst.pop(0)


        # If all available positions were closed, put out Orders at maximum profit
        if not buy_lst and pos_limit - position - buy_quantity > 0:
            bid_amount = pos_limit - position - buy_quantity
            orders.append(Order(product,fairprice - 2,bid_amount))

        if not sell_lst and pos_limit + position - sell_quantity > 0:
            ask_amount = pos_limit + position - sell_quantity
            orders.append(Order(product,fairprice + 2,-ask_amount))

        if buy_lst:
            # Determine the maximum bid after we executed all profitable trades
            bid_max = max(buy_lst)

            # Place competitive bids
            if bid_max < fairprice - 1 and pos_limit - position - buy_quantity > 0:
                bid_price = bid_max + 1
                bid_amount = pos_limit - position - buy_quantity
                orders.append(Order(product,bid_price,bid_amount))

        if sell_lst:
            # Determine the minimum ask after we executed all profitable trades
            ask_min = min(sell_lst)
            
            # Place competitive asks
            if ask_min > fairprice + 1 and pos_limit + position - sell_quantity > 0:
                ask_price = ask_min - 1
                ask_amount = pos_limit + position - sell_quantity
                orders.append(Order(product,ask_price,-ask_amount))
        
        # Buy or sell at fairprice if the position is too close to the limits
        if pos_limit - dump_range <= position <= pos_limit:
            sell_amount = dump_amount
            orders.append(Order(product,fairprice,-sell_amount))
            sell_quantity += sell_amount

        if -pos_limit <= position <= -pos_limit + dump_range:
            buy_amount = dump_amount
            orders.append(Order(product,fairprice,buy_amount))
            buy_quantity += buy_amount

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

        # result["RAINFOREST_RESIN"] = self.resin_ord(state)    
        result["KELP"] = self.kelp_ord(state) 
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
		# Sample conversion request.
        conversions = 1

        return result, conversions, traderData