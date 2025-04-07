from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List
import jsonpickle
import math
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

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

        return value[: max_length - 3] + "..."

logger = Logger()

class Trader:
    def __init__(self):
        self.buy_order_volume = 0
        self.sell_order_volume = 0

    def set_context(self, order_depth: OrderDepth, fair_value: float, position: int, position_limit: int, product: str):
        self.order_depth = order_depth
        self.fair_value = fair_value
        self.position = position
        self.position_limit = position_limit
        self.product = product

    def avellaneda_stoikov_quotes(self, S: float, q: int, T: float, gamma: float, sigma: float, k: float) -> (float, float):
        reservation_price = S - q * gamma * sigma ** 2 * T
        spread = (gamma * sigma ** 2 * T / 2) + (1 / gamma) * math.log(1 + (gamma / k))
        bid = reservation_price - spread
        ask = reservation_price + spread
        return bid, ask

    def kelp_orders(self, state: TradingState, traderObject) -> List[Order]:
        orders: List[Order] = []

        if "T_end" not in traderObject:
            traderObject["T_end"] = 99900

        T_end = traderObject["T_end"]
        t_now = state.timestamp
        T = max((T_end - t_now) / T_end, 0.001)

        gamma = 0.1

        sigma = 0.8
        k = 1

        bid, ask = self.avellaneda_stoikov_quotes(self.fair_value, self.position, T, gamma, sigma, k)

        order_size = 5
        soft_position_limit = 50

        buy_quantity = order_size if self.position < soft_position_limit else 0
        sell_quantity = order_size if self.position > -soft_position_limit else 0

        if buy_quantity > 0:
            orders.append(Order("KELP", round(bid), buy_quantity))
        if sell_quantity > 0:
            orders.append(Order("KELP", round(ask), -sell_quantity))

        return orders

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            adverse_volume = 12
            beta = -0.19

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            filtered_ask = [price for price in order_depth.sell_orders if abs(order_depth.sell_orders[price]) >= adverse_volume]
            filtered_bid = [price for price in order_depth.buy_orders if abs(order_depth.buy_orders[price]) >= adverse_volume]

            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None

            if mm_ask is None or mm_bid is None:
                mmmid_price = (best_ask + best_bid) / 2 if traderObject.get("kelp_last_price") is None else traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price") is not None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * beta
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price

            traderObject["kelp_last_price"] = mmmid_price
            return fair

        return None

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        traderObject = {}

        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)

        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            fair_value_for_kelp = self.kelp_fair_value(state.order_depths["KELP"], traderObject)
            self.set_context(state.order_depths["KELP"], fair_value_for_kelp, kelp_position, 50, "KELP")
            kelp_orders = self.kelp_orders(state, traderObject)
            result["KELP"] = kelp_orders

        trader_data = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data