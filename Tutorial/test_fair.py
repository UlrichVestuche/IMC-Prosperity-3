from Round_2.Gleb.datamodel import OrderDepth, UserId, TradingState, Order
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


        result['RAINFORERST_RESIN'] = []
        if 'KELP' not in state.position.keys() or state.position.get('KELP', 0) == 0:
            result['KELP'] = [Order("KELP",  2025, 1)]
        else:
            result['KELP'] = []

        conversions = 1

        return result, conversions, traderData