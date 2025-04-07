from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np

class Trader:
    def init(self):
        pass

    def run(self, state: TradingState):
        result = {}
        traderData = ""


        #result['AMETHYSTS'] = []
        if 'SQUID_INK' not in state.position.keys() or state.position.get('SQUID_INK', 0) == 0:
            result['SQUID_INK'] = [Order("SQUID_INK",  2005, 1)]
        else:
            result['SQUID_INK'] = []

        conversions = 1

        return result, conversions, traderData