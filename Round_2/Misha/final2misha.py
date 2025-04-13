from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

#########################
# Global variables
# len_long1 = 900
z_max1 = 1.8

len_long1 = 90000


# len_long2 = 300
len_long2 = 90000
z_max2 =  1.5

STD_SQUID_INK_PREV_DAY = 10.7

price_spread = {'PICNIC_BASKET1': 2,'PICNIC_BASKET2': 1,'CROISSANTS': 2,'JAMS': 1,'DJEMBES': 1}
price_max_pb1 = 4


#########################

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
        self.std = 1.1
        self.price_dif  = 0
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
        #self.clear_position_order(orders)

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
            if bbbf!= fair_value:
            #     orders.append(Order("RAINFOREST_RESIN", bbbf, buy_quantity))
            # else:
            #     
                orders.append(Order("RAINFOREST_RESIN", bbbf + insert, buy_quantity))

        sell_quantity = self.position_limit + (self.position - self.sell_order_volume)
        if sell_quantity > 0:
            # Append a sell order. Adjust the sell price based on the best ask and a predefined offset
            if baaf!= fair_value:
            #     orders.append(Order("RAINFOREST_RESIN", baaf, -sell_quantity))
            # else:
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
                    if self.order_depth.sell_orders[best_ask] == 0:
                           del self.order_depth.sell_orders[best_ask]
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
                    if self.order_depth.buy_orders[best_bid] == 0:
                           del self.order_depth.buy_orders[best_bid]
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
            clear_quantity = min(clear_quantity, position_after_take)
            # clear_quantity = position_after_take
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
            clear_quantity = min(clear_quantity, abs(position_after_take))
            # clear_quantity = abs(position_after_take)
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
            #beta = -0.2143
            beta = 0
            

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
   
    def kelp_orders(self) -> List[Order]:
        orders: List[Order] = []
        # Process market orders for KELP using the instrument parameter
        #self.clear_position_order(orders)
        #self.process_market_orders(orders, instrument="KELP")

        #self.clear_position_order(orders)


        soft_position_limit = 50
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
            buy_quantity -= 3
        if self.position < -soft_position_limit:
            sell_quantity -=3
        

        if buy_quantity > 0:
            orders.append(Order("KELP", bid, buy_quantity))
            logger.print('Trying to market make with bid:', bid,"buy quantity:", buy_quantity)
        if sell_quantity > 0:
            orders.append(Order("KELP", ask, -sell_quantity))
            logger.print('Trying to market make with sell:', ask,"sell quantity:", sell_quantity)
        return orders
    


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

           
            if traderObject.get("squid_ink_last_price", None) is not None:
                # Compute last returns as a percentage change for volatility updates
                last_price = traderObject["squid_ink_last_price"]
               
                last_avg_price = traderObject["squid_ink_last_avg_price"]
                

                # Compute raw trend as the absolute difference
                raw_trend = mmmid_price
                alphabar = 1/180
                alpha = 1 - alphabar
                smoothed_avg_price = alpha * last_avg_price + (1 - alpha) * raw_trend
                
                price_dif = (mmmid_price - smoothed_avg_price) 
                
            else:
                smoothed_avg_price = mmmid_price
                price_dif = 0
            alphabar = 1/180
            GARCH_BETA = 1 - alphabar
            
            if 'vol_std' not in traderObject:
                traderObject['vol_std'] = STD_SQUID_INK_PREV_DAY
            else:
                traderObject['vol_std'] = math.sqrt((1-GARCH_BETA) * (price_dif ** 2) + GARCH_BETA * (traderObject['vol_std'] ** 2))
            # Store the current price for the next cycle
            traderObject["squid_ink_last_price"] = mmmid_price
            traderObject["squid_ink_last_avg_price"] = smoothed_avg_price
            traderObject["squid_ink_price_dif"] = price_dif
            # Also store the trend in self for later use in order adjustments
            # Use a sensitivity factor to adjust fair value based on trend
            # sensitivity = 0.0# for now
            # fair = mmmid_price + (sensitivity * smoothed_trend)
            traderObject["fair_value"] = smoothed_avg_price
            self.price_dif = traderObject["squid_ink_price_dif"]  
            self.std = traderObject['vol_std']
            #self.fair_value = smoothed_avg_price

            return smoothed_avg_price
        return None
    

    def ink_orders(self) -> List[Order]:
        orders: List[Order] = []
        
        # Initialize volatility standard deviation if not already defined in traderObject
        logger.print('price_dif:', self.price_dif, 'std:', self.std)
        Z_score = self.price_dif / self.std
        

        # Calculate quantities based on current position and order volumes
        adverse_volume = 12

        filtered_ask = [
            price
            for price in self.order_depth.sell_orders.keys()
            if abs(self.order_depth.sell_orders[price]) >= adverse_volume
        ]
        filtered_bid = [
            price
            for price in self.order_depth.buy_orders.keys()
            if abs(self.order_depth.buy_orders[price]) >= adverse_volume
        ]
        mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
        mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
         
        if mm_ask is None or mm_bid is None:
            return orders

        if Z_score > 1.2 and self.position >-35:
            orders.append(Order("SQUID_INK", math.ceil(self.fair_value) , -15))
            logger.print('Trying to market make with sell:',mm_bid, "sell quantity:", 15)
        elif Z_score < -1.2 and self.position <35:
            orders.append(Order("SQUID_INK", math.floor(self.fair_value), +15))
            logger.print('Trying to market make with buy:', mm_ask, "buy quantity:", 15)
        # if self.position >= -1 and Z_score > 1.5:
        #     orders.append(Order("SQUID_INK", mm_bid, -30))
        #     logger.print('Trying to market make with sell:',mm_bid, "sell quantity:", 20)
        # elif self.position <= 1 and Z_score < -1.5:
        #     orders.append(Order("SQUID_INK", mm_ask, +30))
        #     logger.print('Trying to market make with buy:', mm_ask, "buy quantity:", 20)
        
        
      
        logger.print('Z_score:', Z_score)
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
    # Insert this new method before the definition of run(self, state: TradingState)
    
    def r2_mprice(self, state: TradingState):
        #Compute the weighted mid price for all new products in Round 2

        products = ["PICNIC_BASKET1","PICNIC_BASKET2","CROISSANTS","JAMS","DJEMBES"]
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
        #win_stdev1 = dct['stdev1']

        win_average2 = dct['average2']
        #win_stdev2 = dct['stdev2']

        win_average1 = (1 - alpha1) * win_average1 + alpha1 * spread_price1
        #win_stdev1 = math.sqrt((1 - alpha1) * win_stdev1**2 + alpha1 * (spread_price1 - win_average1)**2)

        win_average2 = (1 - alpha2) * win_average2 + alpha2 * spread_price2
        #win_stdev2 = math.sqrt((1 - alpha2) * win_stdev2**2 + alpha2 * (spread_price2 - win_average2)**2)

        # if win_stdev1:
        #     z_val1 = (spread_price1 - win_average1) / win_stdev1
        # else:
        #     z_val1 = 0

        # if win_stdev2:
        #     z_val2 = (spread_price2 - win_average2) / win_stdev2
        # else:
        #     z_val2 = 0

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
        global z_max1
        global z_max2
        global price_spread
        global max_factor
        global pb1_cutoff
        global pb2_cutoff

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

        # # Decode the dictionary with the spread
        # if state.traderData:
        #     dict_spreads_prev = jsonpickle.decode(state.traderData)
        # else:

        if traderObject.get("zscore1", None) is None:
            traderObject['zscore1'] = 0
            traderObject['average1'] = 0
            traderObject['stdev1'] = 0
            traderObject['zscore2'] = 0
            traderObject['average2'] = 0
            traderObject['stdev2'] = 0
            
        #dict_spreads_prev = traderObject

        # Update the dictionary with the current state
        # dict_spreads = self.pb_spread(state, traderObject)
        # traderObject = self.pb_spread(state, traderObject)
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
        # z_val1 = dict_spreads['zscore1']
        # z_val2 = dict_spreads['zscore2']
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


        ## Starting values for the quantities
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
        # Main execution method called during each trading cycle
        result = {}
        traderObject = {}
        conversions = 1
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        #3k is the best result:)
        

       
    
        PB_dict = {}
        # Check if RAINFOREST_RESIN is available in the current market data
        if "RAINFOREST_RESIN" in state.order_depths:
            # # Get current position for RAINFOREST_RESIN, defaulting to 0 if not present
            # resin_position = state.position["RAINFOREST_RESIN"] if "RAINFOREST_RESIN" in state.position else 0
            # # # Set up the trading context with the latest market data and parameters
            # self.set_context(state.order_depths["RAINFOREST_RESIN"], 10000, 2, resin_position, 50, "RAINFOREST_RESIN")
            # # # Generate trading orders for RAINFOREST_RESIN
            # resin_orders = self.resin_orders()
            # result["RAINFOREST_RESIN"] = resin_orders
            result["RAINFOREST_RESIN"] = self.resin_ord(state) 


        if "KELP" in state.order_depths:
            kelp_position = state.position["KELP"] if "KELP" in state.position else 0
            # Calculate fair price for KELP using the new function
            fair_value_for_kelp = self.kelp_fair_value(state.order_depths["KELP"], traderObject)
            self.set_context(state.order_depths["KELP"], fair_value_for_kelp, 2, kelp_position, 50, 'KELP')
            kelp_orders = self.kelp_orders()
            result["KELP"] = kelp_orders
        
        if "SQUID_INK" in state.order_depths:
            
            squid_ink_position = state.position["SQUID_INK"] if "SQUID_INK" in state.position else 0
            # Calculate fair price for SQUID_INK using the new function
            last_price = traderObject["squid_ink_last_price"] if traderObject.get("squid_ink_last_price", None) is not None else None
            fair_value_for_squid_ink = self.squid_ink_fair_value(state.order_depths["SQUID_INK"], traderObject)
            self.set_context(state.order_depths["SQUID_INK"], fair_value_for_squid_ink, 2, squid_ink_position, 50, 'SQUID_INK')
            squid_ink_orders = self.ink_orders()
            
            result["SQUID_INK"] = squid_ink_orders 
            # if state.timestamp < 5000:
            #     result["SQUID_INK"] = []

        if "PICNIC_BASKET2" in state.order_depths:
            PB_dict = self.pb_ord(state, traderObject)

            result["PICNIC_BASKET1"] = PB_dict['pb1']
            result["PICNIC_BASKET2"] = PB_dict['pb2']
            result["CROISSANTS"] = PB_dict['cro']
            result["JAMS"] = PB_dict['jam']
            result["DJEMBES"] = PB_dict['djem']

            logger.print(PB_dict['pb2'])
        
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        #traderData = jsonpickle.encode(PB_dict['Dict_Spreads'])
        # logger.print("position:",self.position)
        # traderData and conversions could be used for logging or further processing
        
        traderData = jsonpickle.encode(traderObject)
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData