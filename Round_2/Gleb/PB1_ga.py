from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import math
import jsonpickle

class Trader:

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
    
    def pb1_spread(self, state: TradingState, dct):
        #product = "PICNIC_BASKET1"

        # window lengths
        len_long = 5
        len_short = 1

        pb1_mprice = self.r2_mprice(state)["PICNIC_BASKET1"]
        cro_mprice = self.r2_mprice(state)["CROISSANTS"]
        jam_mprice = self.r2_mprice(state)["JAMS"]
        djem_mprice = self.r2_mprice(state)["DJEMBES"]

        synth_mprice = 6 * cro_mprice + 3 * jam_mprice + djem_mprice
        spread_price = pb1_mprice - synth_mprice


        lst_long = dct['long']
        lst_short = dct['short']

        if len(lst_long) < len_long:
            lst_long.append(spread_price)
        else:
            lst_long.pop(0)
            lst_long.append(spread_price)

        if len(lst_short) < len_short:
            lst_short.append(spread_price)
        else:
            lst_short.pop(0)
            lst_short.append(spread_price)

        mean_long = sum(lst_long) / len(lst_long)
        stdev_sq_long = 0

        for p in lst_long:
            stdev_sq_long += (p - mean_long) ** 2

        stdev_long = math.sqrt(stdev_sq_long/len(lst_long))
        #stdev_long = 1
    
        if stdev_long > 0:
            z_val = (spread_price - mean_long) / stdev_long
        else: 
            z_val = 0

        dct['long'] = lst_long
        dct['short'] = lst_short
        dct['zscore'] = z_val

        return dct

    def pb1_ord(self, state: TradingState):
        #product = "PICNIC_BASKET1"
        pos_limit_pb1 = 60
        pos_limit_cro = 250
        pos_limit_jam = 350
        pos_limit_djem = 60

        orders_pb1 = []
        orders_cro = []
        orders_jam = []
        orders_djem = []

        # Get current position for product, defaulting to 0 if not present
        pos_pb1 = state.position["PICNIC_BASKET1"] if "PICNIC_BASKET1" in state.position else 0
        pos_cro = state.position["CROISSANTS"] if "CROISSANTS" in state.position else 0
        pos_jam = state.position["JAMS"] if "JAMS" in state.position else 0
        pos_djem = state.position["DJEMBES"] if "DJEMBES" in state.position else 0

        # Decode the dictionary with the spread
        if state.traderData:
            dict_spread1_prev = jsonpickle.decode(state.traderData)
        else:
            dict_spread1_prev = {'long': [], 'short': [],'zscore': 0}

        # Update the dictionary with the current state
        dict_spread1 = self.pb1_spread(state, dict_spread1_prev)

        # Define mid prices for all products
        pb1_mprice = self.r2_mprice(state)["PICNIC_BASKET1"]
        cro_mprice = self.r2_mprice(state)["CROISSANTS"]
        jam_mprice = self.r2_mprice(state)["JAMS"]
        djem_mprice = self.r2_mprice(state)["DJEMBES"]

        # Define fair prices
        fprice = {'pb1': pb1_mprice,'cro': cro_mprice,'jam': jam_mprice,'djem': djem_mprice}

        # Define prices just above and just below fair prices
        #above_fprice = {}
        #below_fprice = {}

        #for prod in fprice:
        #    if fprice[prod].is_integer():
        #        above_fprice[prod] = int(fprice[prod])
        #        below_fprice[prod] = int(fprice[prod])
        #    else:
        #        above_fprice[prod] = math.ceil(fprice[prod])
        #        below_fprice[prod] = math.floor(fprice[prod])

        # Import the z value
        z_val = dict_spread1['zscore']

        # Define the bid factors
        bid_amount_pb1 = pos_limit_pb1 - pos_pb1
        bid_amount_cro = pos_limit_cro - pos_cro
        bid_amount_jam = pos_limit_jam - pos_jam
        bid_amount_djem = pos_limit_djem - pos_djem

        ask_amount_pb1 = pos_limit_pb1 + pos_pb1
        ask_amount_cro = pos_limit_cro + pos_cro
        ask_amount_jam = pos_limit_jam + pos_jam
        ask_amount_djem = pos_limit_djem + pos_djem

        bid_factor_pb1 = int(bid_amount_pb1 / 10)
        bid_factor_cro = int(bid_amount_cro / 6)
        bid_factor_jam = int(bid_amount_jam / 3)
        bid_factor_djem = int(bid_amount_djem)

        ask_factor_pb1 = int(ask_amount_pb1 / 10)
        ask_factor_cro = int(ask_amount_cro / 6)
        ask_factor_jam = int(ask_amount_jam / 3)
        ask_factor_djem = int(ask_amount_djem)

        bid_factor_pb1 = min(bid_factor_pb1, ask_factor_cro, ask_factor_jam, ask_factor_djem)
        ask_factor_pb1 = min(ask_factor_pb1, bid_factor_cro, bid_factor_jam, bid_factor_djem)

        # Determine the bid price depending on the needed quantity
        ratios = {'pb1': 10,'cro': 6,'jam': 3,'djem': 1}

        for prod in fprice:
            buy_ord = state.order_depths[prod].buy_orders
            sell_ord = state.order_depths[prod].sell_orders

            if prod == 'pb1':
                bid_qneed = ratios[prod] * bid_factor_pb1
                ask_qneed = ratios[prod] * ask_factor_pb1
            else:
                bid_qneed = ratios[prod] * ask_factor_pb1
                ask_qneed = ratios[prod] * bid_factor_pb1

            for p in buy_ord:
                if p > fairprice and pos_limit + position - sell_quantity > 0:
                    sell_amount = min(buy_ord[p], pos_limit + position - sell_quantity)
                    orders.append(Order(product,p,-sell_amount))
                    sell_quantity += sell_amount
                    if sell_amount < buy_ord[p]:
                        buy_lst.append(p)
                else:
                    buy_lst.append(p)


        if z_val > 1 and ask_factor_pb1 > 0:
            orders_pb1.append(Order("PICNIC_BASKET1", above_fprice['pb1'], (-10) * ask_factor_pb1))
            orders_cro.append(Order("CROISSANTS", below_fprice['cro'], 6 * ask_factor_pb1))
            orders_jam.append(Order("JAMS", below_fprice['jam'], 3 * ask_factor_pb1))
            orders_djem.append(Order("DJEMBES", below_fprice['djem'], 1 * ask_factor_pb1))
        elif z_val < -1 and bid_factor_pb1 > 0:
            orders_pb1.append(Order("PICNIC_BASKET1", below_fprice['pb1'], 10 * bid_factor_pb1))
            orders_cro.append(Order("CROISSANTS", above_fprice['cro'], (-6) * bid_factor_pb1))
            orders_jam.append(Order("JAMS", above_fprice['jam'], (-3) * bid_factor_pb1))
            orders_djem.append(Order("DJEMBES", above_fprice['djem'], (-1) * bid_factor_pb1))
        
        return {'pb1': orders_pb1, 'cro': orders_cro,'jam': orders_jam, 'djem': orders_djem, 'Dict_Spread1': dict_spread1}
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        #print("traderData: " + state.traderData)
        #print("Observations: " + str(state.observations))

        result = {}

        PB1_dict = self.pb1_ord(state)

        #result["RAINFOREST_RESIN"] = self.resin_ord(state)
        #result["KELP"] = self.kelp_ord(state, window_lst)
        result["PICNIC_BASKET1"] = PB1_dict['pb1']
        result["CROISSANTS"] = PB1_dict['cro']
        result["JAMS"] = PB1_dict['jam']
        result["DJEMBES"] = PB1_dict['djem']

        print(PB1_dict['cro'])
    
    
        # String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
        traderData = jsonpickle.encode(PB1_dict['Dict_Spread1'])
        

		# Sample conversion request.
        conversions = 1

        return result, conversions, traderData