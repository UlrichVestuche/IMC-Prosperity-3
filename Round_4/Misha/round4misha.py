from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import json
from datamodel import Listing, ConversionObservation, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

#########################
# Global variables
# len_long1 = 900


current_day = 4



z_max1 = 1.8

len_long1 = 9000


# len_long2 = 300
len_long2 = 9000
z_max2 =  1.5

STD_SQUID_INK_PREV_DAY = 14.5

price_spread = {'PICNIC_BASKET1': 2,'PICNIC_BASKET2': 1,'CROISSANTS': 2,'JAMS': 1,'DJEMBES': 1}
price_max_pb1 = 4


#########################

### AUXILIARY FUNCTIONS

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
                observation.sugarPrice,
                observation.sunlightIndex,
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



from math import log, sqrt, exp
from statistics import NormalDist


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        ## changed return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) 

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=500, tolerance=1e-8
    ):
        low_vol = 10**-8
        high_vol = 10**-4
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility
#########################


 
class Rock():

    def __init__(self, position = 0, state: TradingState = None):
        self.position = position
        self.position_limit = 400
        self.state = state
        self.fair_value = 0
        self.width = 1
        self.voucher_strikes = {'VOLCANIC_ROCK_VOUCHER_9500': 9500,
        'VOLCANIC_ROCK_VOUCHER_9750': 9750,
        'VOLCANIC_ROCK_VOUCHER_10000': 10000,
        'VOLCANIC_ROCK_VOUCHER_10250': 10250,
        'VOLCANIC_ROCK_VOUCHER_10500': 10500}
        self.voucher_limit = 200
    def get_position(self):
        dictionary = {}
        for symbol in self.voucher_strikes.keys():
            dictionary[symbol] = self.state.position.get(symbol, 0)
        return dictionary
    def get_mid_price(
        self, rock_voucher: OrderDepth, traderData: dict[str, Any], name: str
    ):
        if (
            len(rock_voucher.buy_orders) > 0
            and len(rock_voucher.sell_orders) > 0
        ):
            best_bid = max(rock_voucher.buy_orders.keys())
            best_ask = min(rock_voucher.sell_orders.keys())
            traderData["prev_" + name + "_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_" + name + "_price"]   
    def get_rock_vouchers_all(self, traderData: dict[str, Any]):
        return [self.get_mid_price(self.state.order_depths[symbol], traderData, symbol) for symbol in self.voucher_strikes.keys()]
    def IV(self, traderData: dict[str, Any]):
        St = self.get_mid_price(self.state.order_depths["VOLCANIC_ROCK"], traderData, "VOLCANIC_ROCK")
        IV_dict = {}
        time_day = 10000 * 100
        T = 7 * time_day
        time_to_expiry = T - 10000 * 100 * current_day - self.state.timestamp
        import math
        sqrt_time_to_expiry = math.sqrt(time_to_expiry)
        for symbol in self.voucher_strikes.keys():
            K = self.voucher_strikes[symbol]
            
           
            voucher_price = self.get_mid_price(self.state.order_depths[symbol], traderData, symbol)
            IV_dict[symbol + "_IV"] = BlackScholes.implied_volatility(voucher_price, St, K, time_to_expiry)
            
        # Fit a parabola to the 5 implied volatility points using transformed x-axis units
        residuals = None
        # Transform each voucher strike into the x-axis unit: log(St/K) / sqrt(time_to_expiry)

        

        x_points = [
            math.log(self.voucher_strikes[symbol]/St) / sqrt_time_to_expiry
            for symbol in self.voucher_strikes.keys()
        ]
        y_points = [IV_dict[symbol + "_IV"] for symbol in self.voucher_strikes.keys()]

        if any(abs(v) <= 0.7* 10**-5 for v in y_points):
            ##logger.print("Skipping points with low IV")
            return None, None

        time_window = 100


        # if (0 in y_points) or (10**-7 in y_points):
        #     ##logger.print("Skipping parabola fit due to invalid IV point(s)")
        # else:
        #     coeffs= self.fit_iv_parabola(x_points, y_points) 
        #     a = coeffs[0]
        #     b = coeffs[1]
        #     c = coeffs[2]
        #     center = -b / (2 * a) if a != 0 else None
        #     ##logger.print("Parabola fit: a =", a, ", center =", center, 'Volatility at center:', a*center**2 + b*center + c, 'Residuals:', residuals)
        #     residuals = ((y_points - np.polyval(coeffs, x_points)) / y_points).tolist()
        #     ##logger.print("Residuals:", residuals)



        if traderData.get('parabola_x', None) is None:
            traderData['parabola_x'] = x_points
            traderData['parabola_y'] = y_points
        elif len(traderData['parabola_x']) >= 5 * time_window:
            
            # Only use the last 5 * time_window points for fitting
            x_sample = traderData['parabola_x'][-5 * time_window:]
            y_sample = traderData['parabola_y'][-5 * time_window:]

            coeffs = self.fit_iv_parabola(x_sample, y_sample)
            a = coeffs[0]
            b = coeffs[1]
            c = coeffs[2]
            center = -b / (2 * a) if a != 0 else None
            ##logger.print("Parabola fit: a =", a, ", center =", center, 'Volatility at center:', a * center**2 + b * center + c)
            traderData['parabola_coeffs'] = coeffs.tolist()

            traderData['parabola_x'] = x_points
            traderData['parabola_y'] = y_points
        else:
            traderData['parabola_x'].extend(x_points)
            traderData['parabola_y'].extend(y_points)
            
        if traderData.get('parabola_coeffs', None) is not None:
            x_arr = np.array(x_points)
            y_arr = np.array(y_points)
            residuals = ((y_arr - np.polyval(traderData['parabola_coeffs'], x_arr)) / y_arr).tolist()
            ##logger.print("Residuals:", residuals)
        else:
            residuals = None
        # ##logger.print(x_points, y_points)
        # ##logger.print(traderData['parabola_x'])
        # ##logger.print(traderData['parabola_y'])
        
        return IV_dict, residuals
    # Insert the following new method in the Rock class (for example, just before the IV method):
    def fit_iv_parabola(self, x_points, y_points):
        x_points = np.array(x_points)
        y_points = np.array(y_points)
        
        # Skip the fit if any y point is 0 or too close to 0 (as before)
        if (0 in y_points) or (10**-7 in y_points):
            ##logger.print("Skipping parabola fit due to invalid IV point(s)")
            return None
        
        # Construct the design matrix for the model f(x) = a * x^2 + c.
        # We only use x^2 and a constant (no x term), ensuring the curve passes through m_t = 0.
        M = np.vstack([x_points**2, np.ones(len(x_points))]).T
        
        # Solve the least squares problem to obtain coefficients [a, c]
        coeffs, residuals, rank, s = np.linalg.lstsq(M, y_points, rcond=None)
        a, c = coeffs
        
        # Return the full quadratic coefficients, in the form [a, 0, c]
        # so that f(x) = a*x^2 + 0*x + c.
        return np.array([a, 0.0, c])
    # def fit_iv_parabola(self, x_points, y_points):
    #     ##logger.print(x_points)
    #     ##logger.print(y_points)
    #     x_points =  np.array(x_points)
    #     y_points =  np.array(y_points)
        
    #     if (0 in y_points) or (10**-7 in y_points):
    #         #logger.print("Skipping parabola fit due to invalid IV point(s)")
    #         return None
    #     coeffs = np.polyfit(x_points, y_points, 2)
    #     return coeffs
    
    def delta_vouchers(self,positions, IV_dict, St, time_to_expiry, Portfolio):
        delta = 0
        for symbol in self.voucher_strikes.keys():
            delta_voucher = BlackScholes.delta(St, self.voucher_strikes[symbol], time_to_expiry, IV_dict[symbol + "_IV"])
            #logger.print('delta_voucher', delta_voucher)
            delta += delta_voucher * (positions[symbol]+Portfolio.get(symbol, 0))
        return delta
    
    def hedge_rock(self, positions, IV_dict, St, time_to_expiry, Portfolio):
        delta = self.delta_vouchers(positions, IV_dict, St, time_to_expiry, Portfolio)
        delta_rock = 1
        delta_hedge = delta + delta_rock * self.state.position.get("VOLCANIC_ROCK", 0)
        delta_hedge_volume = int(delta_hedge)
        return delta_hedge_volume
        
    # def buy_atm_call(self, traderData: dict[str, Any], volume: int = None) -> dict[str, list[Order]]:
    #     """
    #     Buy an ATM call option voucher and hold it.
    #     """
    #     # Get current underlying mid price
    #     St = self.get_mid_price(self.state.order_depths["VOLCANIC_ROCK"], traderData, "VOLCANIC_ROCK")
    #     # Identify ATM strike symbol (closest strike to St)
    #     atm_symbol = min(self.voucher_strikes.items(), key=lambda kv: abs(kv[1] - St))[0]
    #     # Calculate desired volume: fill up to voucher_limit
    #     current_pos = self.state.position.get(atm_symbol, 0)
    #     desired = self.voucher_limit - current_pos
    #     vol = volume if volume is not None else desired
    #     if vol <= 0:
    #         return {}
    #     # Place buy order at best ask
    #     order_depth = self.state.order_depths[atm_symbol]
    #     if not order_depth.sell_orders:
    #         return {}
    #     best_ask = min(order_depth.sell_orders.keys())
    #     size = min(vol, order_depth.sell_orders[best_ask])
    #          # record entry price for exit logic
    #     if 'atm_entry_price' not in traderData:
    #         traderData['atm_entry_price'] = best_ask
    #     return {atm_symbol: [Order(atm_symbol, best_ask, size)]}
    # def exit_atm_call(self, traderData: dict[str, Any], profit_target: float = 1.5) -> dict[str, list[Order]]:
    #     """
    #     Exit the ATM call when profit target is reached.
    #     """
    #     entry = traderData.get('atm_entry_price')
    #     # find ATM symbol based on current mid price
    #     St = self.get_mid_price(self.state.order_depths["VOLCANIC_ROCK"], traderData, "VOLCANIC_ROCK")
    #     atm_symbol = min(self.voucher_strikes.items(), key=lambda kv: abs(kv[1] - St))[0]
    #     pos = self.state.position.get(atm_symbol, 0)
    #     if entry is None or pos <= 0:
    #         return {}
    #     order_depth = self.state.order_depths[atm_symbol]
    #     if not order_depth.buy_orders:
    #         return {}
    #     best_bid = max(order_depth.buy_orders.keys())
    #     # exit when bid price exceeds target
    #     if best_bid >= entry * profit_target:
    #         traderData.pop('atm_entry_price', None)
    #         return {atm_symbol: [Order(atm_symbol, best_bid, -pos)]}
    #     return {}
    def rock_orders(self, traderData: dict[str, Any]):
        soft_limit = 15
        res = 0.04
        exit = 0.01

        delta_upper_limit = 0.9
        delta_lower_limit = 0.1
        
        positions = self.get_position()
        prices  = self.get_rock_vouchers_all(traderData)
    
            
        IV_dict, residuals = self.IV(traderData)
        orders = {}
        if traderData.get('parabola_coeffs', None) is not None:
            logger.print(traderData['parabola_coeffs'])
        if residuals is None:
            return {}
        Portfolio = {}
        St = self.get_mid_price(self.state.order_depths["VOLCANIC_ROCK"], traderData, "VOLCANIC_ROCK")
        time_day = 10000 * 100 
        T = 7 * time_day
        time_to_expiry = T - 10000 * 100 * current_day - self.state.timestamp

        for i, symbol in enumerate(self.voucher_strikes.keys()):
            if IV_dict is not None:
                delta_voucher = BlackScholes.delta(St, self.voucher_strikes[symbol], time_to_expiry, IV_dict[symbol + "_IV"])
            if IV_dict is not None and (delta_voucher < delta_lower_limit or delta_voucher > delta_upper_limit):
                non_trade = True
                # if positions[symbol] > 0:
                #     if self.state.order_depths[symbol].buy_orders:
                #         sell_price = max(self.state.order_depths[symbol].buy_orders.keys())
                #         volume = abs(self.state.order_depths[symbol].buy_orders[sell_price])
                        
                #         sell_volume = min(volume, self.voucher_limit + positions[symbol])
                #         if sell_volume > 0:
                #             orders[symbol] = [Order(symbol, sell_price, -sell_volume)]
                #             Portfolio[symbol] = -sell_volume
                # elif positions[symbol] < 0:
                #     if self.state.order_depths[symbol].sell_orders:
                #         buy_price = min(self.state.order_depths[symbol].sell_orders.keys())
                #         volume = abs(self.state.order_depths[symbol].sell_orders[buy_price])
                        
                #         buy_volume = min(volume, self.voucher_limit - positions[symbol])
                #         if buy_volume > 0:
                #             orders[symbol] = [Order(symbol, buy_price, buy_volume)]
                #             Portfolio[symbol] = buy_volume        
            elif residuals[i] < -res: #and symbol == "VOLCANIC_ROCK_VOUCHER_10000":
                
                if self.state.order_depths[symbol].sell_orders:
                    buy_price = min(self.state.order_depths[symbol].sell_orders.keys())
                    volume = abs(self.state.order_depths[symbol].sell_orders[buy_price])
                    
                    buy_volume = min(volume, self.voucher_limit - positions[symbol])
                    logger.print(symbol, f"Volume: {buy_volume}", f"Position: {positions[symbol]}")
                ## Trade first voucher volume
                    if buy_volume > 0:
                        orders[symbol] = [Order(symbol, buy_price, buy_volume)]
                        Portfolio[symbol] = buy_volume
                        
                else:
                    orders = {}
                    break
                    
                    
            elif residuals[i] > res:# and symbol == "VOLCANIC_ROCK_VOUCHER_10000":
                if self.state.order_depths[symbol].buy_orders:

                    sell_price = max(self.state.order_depths[symbol].buy_orders.keys())
                    volume = abs(self.state.order_depths[symbol].buy_orders[sell_price])
                    sell_volume = min(volume, self.voucher_limit + positions[symbol])
                    logger.print(symbol, f"Volume: {sell_volume}", f"Position: {positions[symbol]}")
                    if sell_volume > 0:
                        orders[symbol] = [Order(symbol, sell_price, -sell_volume)]
                        Portfolio[symbol] = -sell_volume
                else:
                    orders = {}
                    break
            elif residuals[i] < exit:
                if positions[symbol] < 0:
                    if self.state.order_depths[symbol].sell_orders:
                        buy_price = min(self.state.order_depths[symbol].sell_orders.keys())
                        volume = abs(self.state.order_depths[symbol].sell_orders[buy_price])
                    
                        buy_volume = min(volume, self.voucher_limit - positions[symbol])
                        logger.print(symbol, f"Volume: {buy_volume}", f"Position: {positions[symbol]}")
                ## Trade first voucher volume
                    if buy_volume > 0:
                        orders[symbol] = [Order(symbol, buy_price, buy_volume)]
                        Portfolio[symbol] = buy_volume
            elif residuals[i] > -exit:
                if positions[symbol] > 0:
                    if self.state.order_depths[symbol].buy_orders:
                        sell_price = max(self.state.order_depths[symbol].buy_orders.keys())
                        volume = abs(self.state.order_depths[symbol].buy_orders[sell_price])
                        
                        sell_volume = min(volume, self.voucher_limit + positions[symbol])
                        logger.print(symbol, f"Volume: {sell_volume}", f"Position: {positions[symbol]}")
                        if sell_volume > 0:
                            orders[symbol] = [Order(symbol, sell_price, -sell_volume)]
                            Portfolio[symbol] = -sell_volume

       

        if IV_dict is not None:
            delta_hedge_volume = self.hedge_rock(positions, IV_dict, St, time_to_expiry, Portfolio)
            logger.print(f"Delta hedge volume: {delta_hedge_volume}")
            position_rock = self.state.position.get("VOLCANIC_ROCK", 0)
            if delta_hedge_volume < -20:
                delta_hedge_volume = min(delta_hedge_volume, self.position_limit - position_rock)
                if self.state.order_depths["VOLCANIC_ROCK"].sell_orders:
                    buy_price = min(self.state.order_depths["VOLCANIC_ROCK"].sell_orders.keys())
                    orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", buy_price, abs(delta_hedge_volume))]
            elif delta_hedge_volume > 20:
                delta_hedge_volume = min(delta_hedge_volume, self.position_limit + position_rock)
                if self.state.order_depths["VOLCANIC_ROCK"].buy_orders:
                    sell_price = max(self.state.order_depths["VOLCANIC_ROCK"].buy_orders.keys())
                    orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", sell_price, -abs(delta_hedge_volume))]
            elif delta_hedge_volume < 0:
                delta_hedge_volume = min(delta_hedge_volume, self.position_limit - position_rock)
                if self.state.order_depths["VOLCANIC_ROCK"].sell_orders:
                    buy_price = min(self.state.order_depths["VOLCANIC_ROCK"].sell_orders.keys())
                    orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", buy_price-1, abs(delta_hedge_volume))]
            elif delta_hedge_volume > 0:
                delta_hedge_volume = min(delta_hedge_volume, self.position_limit + position_rock)
                if self.state.order_depths["VOLCANIC_ROCK"].buy_orders:
                    sell_price = max(self.state.order_depths["VOLCANIC_ROCK"].buy_orders.keys())
                    orders["VOLCANIC_ROCK"] = [Order("VOLCANIC_ROCK", sell_price+1, -abs(delta_hedge_volume))]         
        
        return orders
    

#########################
class Macarons():
    def __init__(self, position = 0):
        self.position = position
        self.position_limit = 75
        self.conversion_limit = 10
    def implied_bid_ask(
        self,
        observation: ConversionObservation,
    ) -> (float, float):
        return observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1, observation.askPrice + observation.importTariff + observation.transportFees
    
    def mid_price(self, order_depth: OrderDepth):
        return (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys()))/2


    def trends(self, order_depth: OrderDepth, observation: ConversionObservation, traderObject: dict):
        orders =[]
        conversion = 0
        band = 60
        average = 640
        derivative_threshold = 0.05

       

        fair_bid, fair_ask = self.implied_bid_ask(observation)

        if traderObject.get('SunlightIndex10', None) is None:
            traderObject['SunlightIndex10'] = []
            
            
        traderObject['SunlightIndex10'].append(observation.sunlightIndex)
       

        if len(traderObject.get('SunlightIndex10', [])) > 10:
            traderObject['SunlightIndex10'].pop(0)
        
       
        if not traderObject.get('Macarons_Clear', 0):
            traderObject['Macarons_Clear'] = 0
        if len(traderObject.get('SunlightIndex10', [])) == 10:
            traderObject['SunlightIndex10_trend'] = traderObject['SunlightIndex10'][-1] - traderObject['SunlightIndex10'][0]
            

           
            
            if traderObject.get('current_position_Macarons', None) is None:
                traderObject['current_position_Macarons'] = 'arb_make'
            current_position = traderObject['current_position_Macarons']
            #logger.print()
            if observation.sunlightIndex is not None:
                if observation.sunlightIndex > 50:
                    if (self.mid_price(order_depth) - average) > band:
                        current_position = 'short'
                        #if order_depth.buy_orders:
                            #sell_price = max(order_depth.buy_orders.keys())

                        sell_volume = self.position_limit + self.position
                        orders.append(Order("MAGNIFICENT_MACARONS", int(fair_ask), -sell_volume))
                        traderObject['current_position_Macarons'] = 'arb_make' 
                        traderObject['Macarons_Clear'] = 7
                            #self.position += sell_volume
                    elif (self.mid_price( order_depth) - average) < -band and traderObject.get('Macarons_Clear', 0) >0:

                        current_position ='arb_make'
                    #elif self.mid_price( order_depth) < average - band:
                        conversion = min(abs(self.position), 10)
                        traderObject['Macarons_Clear'] -= 1

                        # arb_orders, conversion = self.arb_make(order_depth, observation)
                        # orders.extend(arb_orders)
                    elif current_position == 'arb_make':
                        arb_orders, conversion = self.arb_make(order_depth, observation)
                        orders.extend(arb_orders)
                elif observation.sunlightIndex < 50:
                    current_position = 'arb_make'
                    if traderObject['SunlightIndex10_trend'] > derivative_threshold:
                        if order_depth.buy_orders:
                            #sell_price = max(order_depth.buy_orders.keys())
                            sell_volume = self.position_limit + self.position
                            orders.append(Order("MAGNIFICENT_MACARONS", int(fair_ask), -sell_volume))
                    elif traderObject['SunlightIndex10_trend'] <0:
                        arb_orders, conversion = self.arb_make(order_depth, observation)
                        if arb_orders:
                            orders.extend(arb_orders)
                traderObject['current_position_Macarons'] = current_position
        return orders, conversion
        
    def arb_make(self, order_depth: OrderDepth, observation: ConversionObservation) -> List[Order]:
        # This method creates and returns a list of orders for trading RAINFOREST_RESIN.
        # It uses the market context set previously to decide on market orders and balance orders.
        orders: List[Order] = []

        # Process market orders: evaluate the current order book and add buy/sell orders based on fair value comparisons.
        #self.process_market_orders(orders)

        # Adjust orders based on current position to ensure compliance with position limits.
        #self.clear_position_order(orders)

        # Define a spread to determine acceptable price boundaries for trading decisions.
        spread = 2
        insert = 1
        # take a integer value
        
        bid_price, ask_price = self.implied_bid_ask(observation)

        # Determine the best ask available (baaf) that is above the fair value plus spread, or default to fair value if none exists

        sell_book = [price for price in order_depth.sell_orders.keys() if price >= ask_price + spread]
        if len(sell_book)>1:
            baaf = min(sell_book)
        else:
            baaf = ask_price
        # Determine the best bid available (bbbf) that is below the fair value minus spread, or default to fair value if none exists
        # if self.order_depth.buy_orders:
        #     bbbf = max([price for price in self.order_depth.buy_orders.keys() if price <= fair_value - spread])
        # else:
        #     bbbf = fair_value
        
        ## Calculate the remaining quantity that can be bought or sold based on current position and order volume
        # buy_quantity = self.position_limit - (self.position + self.buy_order_volume)
        # if buy_quantity > 0:
        #     if bbbf!= fair_value:
            
        #         orders.append(Order("RAINFOREST_RESIN", bbbf + insert, buy_quantity))

        #sell_quantity = self.position_limit + (position - self.sell_order_volume)

        # limit = self.conversion_limit
        # limit = 30
        # if abs(self.position)> limit:
        #     sell_quantity =  min(self.position_limit + self.position, 0)
        # else:
        #     
        sell_quantity = min(self.position_limit + self.position, 20)
        #sell_quantity = min(self.position_limit + self.position, abs(30+self.position))
        if sell_quantity > 0:
        
            #orders.append(Order("MAGNIFICENT_MACARONS", baaf - insert, -sell_quantity))
            orders.append(Order("MAGNIFICENT_MACARONS", math.ceil(ask_price), -sell_quantity))
            #orders.append(Order("MAGNIFICENT_MACARONS", math.ceil(ask_price+1), -sell_quantity))
        #logger.print('sell quantity:',sell_quantity,'trying to sell:',baaf, 'try to buy:',ask_price)


        return orders, min(self.conversion_limit, abs(self.position))
    
    def order(self, order_depth: OrderDepth, observation: ConversionObservation, traderObject: dict):
        orders, conversion = self.trends(order_depth, observation, traderObject)
        #orders, conversion = self.arb_make(order_depth, observation)
        return orders,conversion



#########################
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
                    #logger.print('I found good buy price:', best_ask, "fair:",self.fair_value, 'quantity:', quantity)
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
                    #logger.print('I found good sell price:', best_bid, "fair:", self.fair_value, 'quantity:', quantity)
        

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
                
            #logger.print('I tried clearing-selling:',sent_quantity,'with this price : ', fair_for_ask)

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
            #logger.print('I tried clearing-buying:', sent_quantity,'with this price : ', fair_for_bid)

        

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

            ##logger.print(fair)
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
            #logger.print('Trying to market make with bid:', bid,"buy quantity:", buy_quantity)
        if sell_quantity > 0:
            orders.append(Order("KELP", ask, -sell_quantity))
            #logger.print('Trying to market make with sell:', ask,"sell quantity:", sell_quantity)
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
        #logger.print('price_dif:', self.price_dif, 'std:', self.std)
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
            #logger.print('Trying to market make with sell:',mm_bid, "sell quantity:", 15)
        elif Z_score < -1.2 and self.position <35:
            orders.append(Order("SQUID_INK", math.floor(self.fair_value), +15))
            #logger.print('Trying to market make with buy:', mm_ask, "buy quantity:", 15)
        # if self.position >= -1 and Z_score > 1.5:
        #     orders.append(Order("SQUID_INK", mm_bid, -30))
        #     #logger.print('Trying to market make with sell:',mm_bid, "sell quantity:", 20)
        # elif self.position <= 1 and Z_score < -1.5:
        #     orders.append(Order("SQUID_INK", mm_ask, +30))
        #     #logger.print('Trying to market make with buy:', mm_ask, "buy quantity:", 20)
        
        
      
        #logger.print('Z_score:', Z_score)
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
        band = 20
        if spread_price1 - win_average1 > band:
            z_val1 = 30
        elif spread_price1 - win_average1 < -band:
            z_val1 = -30
        else:
            z_val1 = 0

        if spread_price2 - win_average2 > band:
            z_val2 = 30
        elif spread_price2 - win_average2 < -band:
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
            # if state.timestamp > 5000:

            PB_dict = self.pb_ord(state, traderObject)

            result["PICNIC_BASKET1"] = PB_dict['pb1']
            result["PICNIC_BASKET2"] = PB_dict['pb2']
            result["CROISSANTS"] = PB_dict['cro']
            result["JAMS"] = PB_dict['jam']
            result["DJEMBES"] = PB_dict['djem']

            #logger.print(PB_dict['pb2'])
        

        if "VOLCANIC_ROCK_VOUCHER_9500" in state.order_depths:
            rock = Rock(state=state)
            rock_vouchers = rock.rock_orders(traderObject)
            ##logger.print(rock_vouchers)
            if rock_vouchers.get("VOLCANIC_ROCK_VOUCHER_9500", None) is not None:
                result["VOLCANIC_ROCK_VOUCHER_9500"] = rock_vouchers["VOLCANIC_ROCK_VOUCHER_9500"]
            if rock_vouchers.get("VOLCANIC_ROCK_VOUCHER_9750", None) is not None:
                result["VOLCANIC_ROCK_VOUCHER_9750"] = rock_vouchers["VOLCANIC_ROCK_VOUCHER_9750"]
            if rock_vouchers.get("VOLCANIC_ROCK_VOUCHER_10000", None) is not None:
                result["VOLCANIC_ROCK_VOUCHER_10000"] = rock_vouchers["VOLCANIC_ROCK_VOUCHER_10000"]
            if rock_vouchers.get("VOLCANIC_ROCK_VOUCHER_10250", None) is not None:
                result["VOLCANIC_ROCK_VOUCHER_10250"] = rock_vouchers["VOLCANIC_ROCK_VOUCHER_10250"]
            if rock_vouchers.get("VOLCANIC_ROCK_VOUCHER_10500", None) is not None:
                result["VOLCANIC_ROCK_VOUCHER_10500"] = rock_vouchers["VOLCANIC_ROCK_VOUCHER_10500"]
            if rock_vouchers.get("VOLCANIC_ROCK", None) is not None:
                result["VOLCANIC_ROCK"] = rock_vouchers["VOLCANIC_ROCK"]
        if "MAGNIFICENT_MACARONS" in state.order_depths:
            position = state.position.get("MAGNIFICENT_MACARONS", 0)
            macarons = Macarons(position)
            conv_obs = state.observations.conversionObservations["MAGNIFICENT_MACARONS"]
            orders, conversions = macarons.order(state.order_depths["MAGNIFICENT_MACARONS"], conv_obs, traderObject)
            result["MAGNIFICENT_MACARONS"] = orders

        #traderData = jsonpickle.encode(PB_dict['Dict_Spreads'])
        # ##logger.print("position:",self.position)
        # traderData and conversions could be used for logging or further processing
        
        traderData = jsonpickle.encode(traderObject)
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData