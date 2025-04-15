from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import math

class Trader:

    def cro_ord(self, state: TradingState) -> List[Order]:
        product = "CROISSANTS"
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        if position == 0:
            orders.append(Order(product,4276,1))

        return orders
    
    def r3_ord(self, state: TradingState) -> List[Order]:
        products = ["VOLCANIC_ROCK","VOLCANIC_ROCK_VOUCHER_9750","VOLCANIC_ROCK_VOUCHER_9500","VOLCANIC_ROCK_VOUCHER_10250","VOLCANIC_ROCK_VOUCHER_10000","VOLCANIC_ROCK_VOUCHER_10500"]
        prices = {"VOLCANIC_ROCK": 10219,"VOLCANIC_ROCK_VOUCHER_9750": 470,"VOLCANIC_ROCK_VOUCHER_9500": 719,"VOLCANIC_ROCK_VOUCHER_10250": 64,"VOLCANIC_ROCK_VOUCHER_10000": 234,"VOLCANIC_ROCK_VOUCHER_10500": 9}
        pos = {}
        orders = {}

        # Get current positions for all products, defaulting to 0 if not present
        for prod in products:
            pos[prod] = state.position[prod] if prod in state.position else 0
            orders[prod] = []

        for prod in products:
            if pos[prod] == 0:
                orders[prod].append(Order(prod,prices[prod],1))

        return orders
    
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))
        
        result = {}
        conversions = 1

        products = ["VOLCANIC_ROCK","VOLCANIC_ROCK_VOUCHER_9750","VOLCANIC_ROCK_VOUCHER_9500","VOLCANIC_ROCK_VOUCHER_10250","VOLCANIC_ROCK_VOUCHER_10000","VOLCANIC_ROCK_VOUCHER_10500"]

        for prod in products:
            result[prod] = self.r3_ord(state)[prod]
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"         

        return result, conversions, traderData