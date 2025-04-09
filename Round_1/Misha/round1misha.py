from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

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
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

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
                observation.sunlight,
                observation.humidity,
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

        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        # Initialize the Trader object with empty lists for starfruit prices and VWAP (volume weighted average price)
        self.buy_order_volume = 0
        self.sell_order_volume = 0
    
    def set_context(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int, product:string):
        # Set up the trading context using provided market data and internal state parameters.
        # This method initializes the order book, fair value, trading width, current position, and position limits.
        self.order_depth = order_depth
        self.fair_value = fair_value
        self.width = width
        self.position = position
        self.position_limit = position_limit
        self.product = product
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
        # take a integer value
        fair_value = int(self.fair_value)

        # Determine the best ask available (baaf) that is above the fair value plus spread, or default to fair value if none exists
        if self.order_depth.sell_orders:
            baaf = min([price for price in self.order_depth.sell_orders.keys() if price >= fair_value + spread])
        else:
            baaf = fair_value
        # Determine the best bid available (bbbf) that is below the fair value minus spread, or default to fair value if none exists
        if self.order_depth.buy_orders:
            bbbf = max([price for price in self.order_depth.buy_orders.keys() if price <= fair_value - spread])
        else:
            bbbf = fair_value

        ## Calculate the remaining quantity that can be bought or sold based on current position and order volume
        buy_quantity = self.position_limit - (self.position + self.buy_order_volume)
        if buy_quantity > 0:
            # Append a buy order at a price slightly above the best bid
            orders.append(Order("RAINFOREST_RESIN", bbbf + insert, buy_quantity))

        sell_quantity = self.position_limit + (self.position - self.sell_order_volume)
        if sell_quantity > 0:
            # Append a sell order. Adjust the sell price based on the best ask and a predefined offset
            if baaf - insert == fair_value:
                orders.append(Order("RAINFOREST_RESIN", baaf, -sell_quantity))
            else:
                orders.append(Order("RAINFOREST_RESIN", baaf - insert, -sell_quantity))

        return orders
    
    def process_market_orders(self, orders: List[Order], instrument: str = "RAINFOREST_RESIN") -> None:
        # Process sell orders: check each sell order in the order book

        width = 1 

        if self.order_depth.sell_orders:
            best_ask = min(self.order_depth.sell_orders.keys())
            ask_amount = -self.order_depth.sell_orders[best_ask]
            if best_ask < self.fair_value:
                quantity = min(ask_amount, self.position_limit - self.position)
                if quantity > 0:
                    orders.append(Order(instrument, best_ask, quantity))
                    self.buy_order_volume += quantity

                    # delete if empty
                    self.order_depth.sell_orders[best_ask] += quantity
                    #if self.order_depth.sell_orders[best_ask] == 0:
                    #        del self.order_depth.sell_orders[best_ask]
                    logger.print('I found good buy price:', best_ask, "fair:",self.fair_value, 'quantity:', quantity)
        # Process buy orders: evaluate if the best bid is favorable
        if len(self.order_depth.buy_orders) != 0:
            best_bid = max(self.order_depth.buy_orders.keys())
            best_bid_amount = self.order_depth.buy_orders[best_bid]
            # If the best bid price is higher than the fair value, consider selling
            if best_bid > self.fair_value:
                # Determine how many units can be sold without exceeding the position limit
                quantity = min(best_bid_amount, self.position_limit + self.position)
                if quantity > 0:
                    orders.append(Order(instrument, best_bid, -1 * quantity))
                    self.sell_order_volume += quantity

                    # delete if empty
                    self.order_depth.buy_orders[best_bid] -= quantity
                    #if self.order_depth.buy_orders[best_bid] == 0:
                    #        del self.order_depth.buy_orders[best_bid]
                    logger.print('I found good sell price:', best_bid, "fair:", self.fair_value, 'quantity:', quantity)
        

    def clear_position_order(self, orders: List[Order]) -> None:
        # Adjust orders to clear or reduce positions if current positions exceed limits
        # Calculate the net position after processing market orders
        position_after_take = self.position + self.buy_order_volume - self.sell_order_volume
        # Round the fair value and determine floor/ceiling for bid/ask decisions
        width = 1
        fair = round(self.fair_value)

        # fair_for_bid = round(fair - width)
        # fair_for_ask = round(fair + width)

        #taking the number just above and just below
        # fair_for_ask = math.ceil(self.fair_value)
        # fair_for_bid = math.floor(self.fair_value)

        fair_for_ask = math.ceil(self.fair_value+1)
        
        fair_for_bid = math.floor(self.fair_value-1)
        # taking the one above
        if math.ceil(self.fair_value) == self.fair_value:
            fair_for_ask = fair+1
            fair_for_bid = fair-1
        

        buy_quantity = self.position_limit - (self.position + self.buy_order_volume)
        sell_quantity = self.position_limit + (self.position - self.sell_order_volume)
        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in self.order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            #clear_quantity = min(clear_quantity, position_after_take)
            clear_quantity = position_after_take
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(self.product, fair_for_ask, -abs(sent_quantity)))
                self.sell_order_volume += abs(sent_quantity)
                
            logger.print('I tried clearing-selling:',sent_quantity,'with this price : ', fair_for_ask)

        if position_after_take < 0:
            #Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in self.order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            #clear_quantity = min(clear_quantity, abs(position_after_take))
            clear_quantity = abs(position_after_take)
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(self.product, fair_for_bid, abs(sent_quantity)))
                self.buy_order_volume += abs(sent_quantity)
            logger.print('I tried clearing-buying:', sent_quantity,'with this price : ', fair_for_bid)

        

        # NOTE: Additional logic to clear positions can be implemented here if needed

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            ### Declare them in class??
            ## Filter and decay
            adverse_volume = 12
            beta = -0.2143
            #beta = 0
            

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= adverse_volume #self.params[Product.STARFRUIT]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= adverse_volume #self.params[Product.STARFRUIT]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * beta #self.params[Product.STARFRUIT]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            traderObject['fair_value'] = fair

            #logger.print(fair)
            return fair
       
        return None
    def squid_ink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            # Filter and decay parameters for SQUID_INK
            adverse_volume = 12
            #beta = -0.2143
            beta = 0
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= adverse_volume
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= adverse_volume
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None

            if mm_ask is None or mm_bid is None:
                if traderObject.get("squid_ink_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["squid_ink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # if traderObject.get("squid_ink_last_price", None) is not None:
            #     last_price = traderObject["squid_ink_last_price"]
            #     last_returns = (mmmid_price - last_price) / last_price
            #     # Record previous returns
            #     if "squid_ink_prev_returns" not in traderObject:
            #         traderObject["squid_ink_prev_returns"] = []
            #     traderObject["squid_ink_prev_returns"].append(last_returns)
            #     traderObject["squid_ink_last_returns"] = last_returns
            #     pred_returns = last_returns * beta
            #     fair = mmmid_price + (mmmid_price * pred_returns)
            # else:
            #     fair = mmmid_price

            # traderObject["squid_ink_last_price"] = mmmid_price
            # traderObject["fair_value"] = fair
            if traderObject.get("squid_ink_last_price", None) is not None:
                # Compute last returns as a percentage change for volatility updates
                last_price = traderObject["squid_ink_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                traderObject["squid_ink_last_returns"] = last_returns

                # Compute raw trend as the absolute difference
                raw_trend = mmmid_price - last_price
                previous_trend = traderObject.get("squid_ink_trend", 0)
                smoothed_trend = 0.9 * previous_trend + 0.1 * raw_trend
                traderObject["squid_ink_trend"] = smoothed_trend
            else:
                last_returns = 0
                smoothed_trend = 0

            # Store the current price for the next cycle
            traderObject["squid_ink_last_price"] = mmmid_price

            # Also store the trend in self for later use in order adjustments
            self.trend = smoothed_trend

            # Use a sensitivity factor to adjust fair value based on trend
            sensitivity = 0.0# for now
            fair = mmmid_price + (sensitivity * smoothed_trend)
            traderObject["fair_value"] = fair


            return fair
        return None
    def kelp_orders(self) -> List[Order]:
        orders: List[Order] = []
        # Process market orders for KELP using the instrument parameter
        self.clear_position_order(orders)
        #self.process_market_orders(orders, instrument="KELP")

        #self.clear_position_order(orders)


        soft_position_limit = 47
        hard_position_limit = 50
        spread = 1.5
        insert = 1
       
        fair_value = self.fair_value
        
        #self.calculate_fair_price(self.order_depth)
        asks_above_fair = [
            price
            for price in self.order_depth.sell_orders.keys()
            if price >= fair_value + spread
        ]
        bids_below_fair = [
            price
            for price in self.order_depth.buy_orders.keys()
            if price <= fair_value - spread
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        if fair_value == round(fair_value):
            ask = round(fair_value + 1)
            bid = round(fair_value - 1)
        else:
            ask = round(fair_value + 0.5)
            bid = round(fair_value - 0.5)

        if best_ask_above_fair != None:
            ask = best_ask_above_fair - insert
            # if abs(best_ask_above_fair - fair_value) <= join_edge:
            #     ask = best_ask_above_fair  # join
            # else:
            #     ask = best_ask_above_fair - 1  # penny

        
        if best_bid_below_fair != None:
            bid = best_bid_below_fair + insert



        RecoverPosition = 3


        # Calculate quantities based on current position and order volumes

        # if self.position > soft_position_limit:
        #     buy_quantity = 0
        #     #self.sell_order_volume +=  RecoverPosition
        # else:
        #     buy_quantity = self.position_limit - (self.position + self.buy_order_volume)
        

        # if self.position < -soft_position_limit:
        #     sell_quantity = 0
        #     #self.buy_order_volume += RecoverPosition
        # else:
        #     sell_quantity = self.position_limit + (self.position - self.sell_order_volume)

        # Let's to clean up slightly buy/sell slighlty higher/lower
        # if self.position > soft_position_limit:
        #     ask -=1
        # elif self.position < -1 * soft_position_limit:
        #     bid += 1

        # sell_quantity = self.position_limit  + (self.position - self.sell_order_volume)
        

        buy_quantity = self.position_limit  - (self.position + self.buy_order_volume)
        sell_quantity = self.position_limit  + (self.position - self.sell_order_volume)
        # buy_quantity = soft_position_limit  - (self.position + self.buy_order_volume)
        # sell_quantity = soft_position_limit  + (self.position - self.sell_order_volume)

        if self.position > soft_position_limit:
            buy_quantity -= 1
        if self.position < -soft_position_limit:
            sell_quantity -=1
        

        if buy_quantity > 0:
            orders.append(Order("KELP", bid, buy_quantity))
            logger.print('Trying to market make with bid:', bid,"buy quantity:", buy_quantity)
        if sell_quantity > 0:
            orders.append(Order("KELP", ask, -sell_quantity))
            logger.print('Trying to market make with sell:', ask,"sell quantity:", sell_quantity)
        return orders

    def ink_orders(self) -> List[Order]:
        orders: List[Order] = []
        #self.process_market_orders(orders, instrument="SQUID_INK") 
        self.clear_position_order(orders) 
        soft_position_limit = 10
        hard_position_limit = 50
        spread = 1.5
        insert = 1
        
        fair_value = self.fair_value
        
        # Identify ask and bid prices relative to fair value
        asks_above_fair = [
            price for price in self.order_depth.sell_orders.keys()
            if price >= fair_value + spread
        ]

        bids_below_fair = [
            price for price in self.order_depth.buy_orders.keys()
            if price <= fair_value - spread
        ]
        
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        
        # Set initial ask and bid based on the fair value
        if fair_value == round(fair_value):
            ask = round(fair_value + 1)
            bid = round(fair_value - 1)
        else:
            ask = round(fair_value + 1)
            bid = round(fair_value - 1)
        
        # Adjust ask and bid based on order book information if available
        if best_ask_above_fair is not None:
            ask = best_ask_above_fair - insert
        if best_bid_below_fair is not None:
            bid = best_bid_below_fair + insert
        
        # RecoverPosition is defined but not used further here
        RecoverPosition = 3

        hard_limit = 10
        
        # Calculate buy and sell quantities based on current positions and order volumes
        buy_quantity = min((self.position_limit - (self.position + self.buy_order_volume)),hard_limit)
        sell_quantity = min((self.position_limit + (self.position - self.sell_order_volume)),hard_limit)

        trend = getattr(self, "trend", 0)
        # Instead of a fixed factor, compute an adjustment factor that scales with trend strength.
        # For instance, you can set a base scaling value and then multiply it by some function of the trend magnitude.
        base_factor = 0.2  # base factor when trend strength is at a minimum threshold
        trend_strength = abs(trend)
        # You might cap trend_strength to avoid excessive scaling:
        max_strength = 0.5  # or any number you find suitable from your data
        normalized_strength = min(trend_strength, max_strength) / max_strength

        # Now the adjustment factor becomes a blend between no adjustment (1.0 means no reduction) and full adjustment (base_factor).
        # For example, if normalized_strength=1 (strong trend), then adjustment_factor = base_factor.
        # If normalized_strength=0, then adjustment_factor = 1 (i.e. no adjustment).
        adjustment_factor = 1.0 - (1.0 - base_factor) * normalized_strength

        # Then adjust:
        if trend > 0:
            sell_quantity = int(sell_quantity * adjustment_factor)
        elif trend < 0:
            buy_quantity = int(buy_quantity * adjustment_factor)
        

        if self.position > soft_position_limit:
            buy_quantity -= 30
        if self.position < -soft_position_limit:
            sell_quantity -= 30
        
        # Create orders for SQUID_INK if there is a positive quantity to trade
        if buy_quantity > 0:
            orders.append(Order("SQUID_INK", bid, buy_quantity))
            logger.print('Trying to market make with bid:', bid, "buy quantity:", buy_quantity)
        if sell_quantity > 0:
            orders.append(Order("SQUID_INK", ask, -sell_quantity))
            logger.print('Trying to market make with sell:', ask, "sell quantity:", sell_quantity)
        
        return orders
    def run(self, state: TradingState):
        # Main execution method called during each trading cycle
        result = {}
        traderObject = {}
        conversions = 1
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        #3k is the best result:)
        


        # Check if RAINFOREST_RESIN is available in the current market data
        # if "RAINFOREST_RESIN" in state.order_depths:
        #     # Get current position for RAINFOREST_RESIN, defaulting to 0 if not present
        #     resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
        #     # Set up the trading context with the latest market data and parameters
        #     self.set_context(state.order_depths["RAINFOREST_RESIN"], 10000, 2, resin_position, 50)
        #     # Generate trading orders for RAINFOREST_RESIN
        #     #resin_orders = self.resin_orders()
        #     #result["RAINFOREST_RESIN"] = resin_orders


        # if "KELP" in state.order_depths:
        #     kelp_position = state.position["KELP"] if "KELP" in state.position else 0
        #     # Calculate fair price for KELP using the new function
        #     fair_value_for_kelp = self.kelp_fair_value(state.order_depths["KELP"], traderObject)
        #     self.set_context(state.order_depths["KELP"], fair_value_for_kelp, 2, kelp_position, 50, 'KELP')
        #     kelp_orders = self.kelp_orders()
        #     result["KELP"] = kelp_orders

        if "SQUID_INK" in state.order_depths:
            squid_ink_position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
            # Calculate fair price for SQUID_INK using the new function
            fair_value_for_squid_ink = self.squid_ink_fair_value(state.order_depths["SQUID_INK"], traderObject)
            # GARCH(1,1) parameters from volatility analysis
            GARCH_ALPHA = 0.0225
            GARCH_BETA = 0.9775

            # Initialize volatility standard deviation if not already defined in traderObject
            if 'vol_std' not in traderObject:
                traderObject['vol_std'] = 0.0009120110012933297  # baseline standard deviation

            # Update volatility forecast using the GARCH(1,1) formula if a last return exists
            if traderObject.get('squid_ink_last_returns', None) is not None:
                last_return = traderObject['squid_ink_last_returns']
                traderObject['vol_std'] = math.sqrt(GARCH_ALPHA * (last_return ** 2) + GARCH_BETA * (traderObject['vol_std'] ** 2))

            # Set dynamic threshold based on the updated volatility forecast
            threshold = 2 * traderObject['vol_std']

            # if traderObject.get("squid_ink_last_returns", None) is not None and abs(traderObject["squid_ink_last_returns"]) > threshold:
            #     logger.print("Skipping SQUID_INK trade due to outlier return:", traderObject["squid_ink_last_returns"])
            #     result["SQUID_INK"] = []
            # else:
            #     self.set_context(state.order_depths["SQUID_INK"], fair_value_for_squid_ink, 2, squid_ink_position, 50, 'SQUID_INK')
            #     squid_ink_orders = self.ink_orders()
            #     result["SQUID_INK"] = squid_ink_orders
            self.set_context(state.order_depths["SQUID_INK"], fair_value_for_squid_ink, 2, squid_ink_position, 50, 'SQUID_INK')
            squid_ink_orders = self.ink_orders()
            result["SQUID_INK"] = squid_ink_orders

        logger.print("position:",self.position)
        # traderData and conversions could be used for logging or further processing
        traderData = jsonpickle.encode(traderObject)
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData