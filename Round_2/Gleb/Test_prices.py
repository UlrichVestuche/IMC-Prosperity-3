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
    
    def jam_ord(self, state: TradingState) -> List[Order]:
        product = "JAMS"
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        if position == 0:
            orders.append(Order(product,6543,1))

        return orders
    
    def pb1_ord(self, state: TradingState) -> List[Order]:
        product = "PICNIC_BASKET1"
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        if position == 0:
            orders.append(Order(product,58715,1))

        return orders
    
    def pb2_ord(self, state: TradingState) -> List[Order]:
        product = "PICNIC_BASKET2"
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        if position == 0:
            orders.append(Order(product,30257,1))

        return orders
    
    def djem_ord(self, state: TradingState) -> List[Order]:
        product = "DJEMBES"
        orders = []

        # Get current position for product, defaulting to 0 if not present
        position = state.position[product] if product in state.position else 0

        if position == 0:
            orders.append(Order(product,13410,1))

        return orders
    
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))
        
        result = {}
   
        result["CROISSANTS"] = self.cro_ord(state)
        result["JAMS"] = self.jam_ord(state)
        result["PICNIC_BASKET1"] = self.pb1_ord(state)
        result["PICNIC_BASKET2"] = self.pb2_ord(state)
        result["DJEMBES"] = self.djem_ord(state)
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
		# Sample conversion request.
        conversions = 1

        return result, conversions, traderData