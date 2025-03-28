from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math

class Trader:
    def __init__(self):
        # Initialize the Trader object with empty lists for starfruit prices and VWAP (volume weighted average price)
        self.starfruit_prices = []
        self.starfruit_vwap = []
    
    def set_context(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int):
        # Set up the trading context using provided market data and internal state parameters.
        # This method initializes the order book, fair value, trading width, current position, and position limits.
        self.order_depth = order_depth
        self.fair_value = fair_value
        self.width = width
        self.position = position
        self.position_limit = position_limit
        # Initialize order volumes for tracking how many units are bought or sold in market orders
        self.buy_order_volume = 0
        self.sell_order_volume = 0

    def resin_orders(self) -> List[Order]:
        # This method creates and returns a list of orders for trading RAINFOREST_RESIN.
        # It uses the market context set previously to decide on market orders and balance orders.
        orders: List[Order] = []

        # Process market orders: evaluate the current order book and add buy/sell orders based on fair value comparisons.
        self.process_market_orders(orders)

        # Adjust orders based on current position to ensure compliance with position limits.
        self.clear_position_order(orders)

        # Define a spread to determine acceptable price boundaries for trading decisions.
        spread = 2
        insert = 1
        # Determine the best ask available (baaf) that is above the fair value plus spread, or default to fair value if none exists
        if self.order_depth.sell_orders:
            baaf = min([price for price in self.order_depth.sell_orders.keys() if price >= self.fair_value + spread])
        else:
            baaf = self.fair_value
        # Determine the best bid available (bbbf) that is below the fair value minus spread, or default to fair value if none exists
        if self.order_depth.buy_orders:
            bbbf = max([price for price in self.order_depth.buy_orders.keys() if price <= self.fair_value - spread])
        else:
            bbbf = self.fair_value

        ## Calculate the remaining quantity that can be bought or sold based on current position and order volume
        buy_quantity = self.position_limit - (self.position + self.buy_order_volume)
        if buy_quantity > 0:
            # Append a buy order at a price slightly above the best bid
            orders.append(Order("RAINFOREST_RESIN", bbbf + insert, buy_quantity))

        sell_quantity = self.position_limit + (self.position - self.sell_order_volume)
        if sell_quantity > 0:
            # Append a sell order. Adjust the sell price based on the best ask and a predefined offset
            if baaf - insert == self.fair_value:
                orders.append(Order("RAINFOREST_RESIN", baaf, -sell_quantity))
            else:
                orders.append(Order("RAINFOREST_RESIN", baaf - insert, -sell_quantity))

        return orders
    
    def process_market_orders(self, orders: List[Order]) -> None:
        # Process sell orders: check each sell order in the order book
        if len(self.order_depth.sell_orders) != 0:
            for ask in self.order_depth.sell_orders.keys():
                # Calculate the available sell quantity (negative value indicates sell orders)
                ask_amount = -1 * self.order_depth.sell_orders[ask]
                # If the ask price is lower than the fair value, consider buying
                if ask < self.fair_value:
                    # Determine how many units can be bought without exceeding the position limit
                    quantity = min(ask_amount, self.position_limit - self.position - self.buy_order_volume)
                    if quantity > 0:
                        orders.append(Order("RAINFOREST_RESIN", ask, quantity))
                        self.buy_order_volume += quantity
        # Process buy orders: evaluate if the best bid is favorable
        if len(self.order_depth.buy_orders) != 0:
            best_bid = max(self.order_depth.buy_orders.keys())
            best_bid_amount = self.order_depth.buy_orders[best_bid]
            # If the best bid price is higher than the fair value, consider selling
            if best_bid > self.fair_value:
                # Determine how many units can be sold without exceeding the position limit
                quantity = min(best_bid_amount, self.position_limit + self.position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                    self.sell_order_volume += quantity

    def clear_position_order(self, orders: List[Order]) -> None:
        # Adjust orders to clear or reduce positions if current positions exceed limits
        # Calculate the net position after processing market orders
        position_after_take = self.position + self.buy_order_volume - self.sell_order_volume
        # Round the fair value and determine floor/ceiling for bid/ask decisions
        fair = round(self.fair_value)
        fair_for_bid = math.floor(self.fair_value)
        fair_for_ask = math.ceil(self.fair_value)

        # Calculate how many more units can be bought or sold without exceeding limits
        buy_quantity = self.position_limit - (self.position + self.buy_order_volume)
        sell_quantity = self.position_limit + (self.position - self.sell_order_volume)

        # NOTE: Additional logic to clear positions can be implemented here if needed

    def run(self, state: TradingState):
        # Main execution method called during each trading cycle
        result = {}

        # Check if RAINFOREST_RESIN is available in the current market data
        if "RAINFOREST_RESIN" in state.order_depths:
            # Get current position for RAINFOREST_RESIN, defaulting to 0 if not present
            resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            # Set up the trading context with the latest market data and parameters
            self.set_context(state.order_depths["RAINFOREST_RESIN"], 10000, 2, resin_position, 50)
            # Generate trading orders for RAINFOREST_RESIN
            resin_orders = self.resin_orders()
            result["RAINFOREST_RESIN"] = resin_orders

        # traderData and conversions could be used for logging or further processing
        traderData = ''
        conversions = 1

        return result, conversions, traderData