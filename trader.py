from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle


class Trader:
    """
    Prosperity 4 — Aggressive v3
    
    Key changes from v2:
    - EMERALDS: Overbid/undercut the existing book instead of fixed offset
    - TOMATOES: Same overbid/undercut approach with Wall Mid as fair value
    - Both: More aggressive taking — take at fair value too, not just better
    - Both: Multi-level quoting to capture more fills
    """

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self.trade_emeralds(state)

        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self.trade_tomatoes(state)

        trader_data = ""
        return result, conversions, trader_data

    def trade_emeralds(self, state: TradingState) -> List[Order]:
        PRODUCT = "EMERALDS"
        FAIR_VALUE = 10_000
        POSITION_LIMIT = 80

        position = state.position.get(PRODUCT, 0)
        order_depth = state.order_depths[PRODUCT]
        orders: List[Order] = []

        # Phase 1: TAKE — aggressively sweep everything at or better than FV
        # Buy anything at or below fair value (not just below)
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= FAIR_VALUE:
                    ask_vol = order_depth.sell_orders[ask_price]
                    buy_qty = min(-ask_vol, POSITION_LIMIT - position)
                    if buy_qty > 0:
                        orders.append(Order(PRODUCT, ask_price, buy_qty))
                        position += buy_qty
                else:
                    break

        # Sell anything at or above fair value
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price >= FAIR_VALUE:
                    bid_vol = order_depth.buy_orders[bid_price]
                    sell_qty = min(bid_vol, POSITION_LIMIT + position)
                    if sell_qty > 0:
                        orders.append(Order(PRODUCT, bid_price, -sell_qty))
                        position -= sell_qty
                else:
                    break

        # Phase 2: MAKE — overbid the best bid, undercut the best ask
        # This is what Frankfurt Hedgehogs did: "placed passive quotes
        # slightly better than any existing liquidity"
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 9992
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 10008

        # Overbid by 1, undercut by 1, but never cross fair value
        our_bid = min(best_bid + 1, FAIR_VALUE - 1)
        our_ask = max(best_ask - 1, FAIR_VALUE + 1)

        # Skew based on inventory
        skew = round(-position * 2 / POSITION_LIMIT)
        our_bid += skew
        our_ask += skew

        # Clamp: never bid above FV-1 or ask below FV+1
        our_bid = min(our_bid, FAIR_VALUE - 1)
        our_ask = max(our_ask, FAIR_VALUE + 1)
        if our_bid >= our_ask:
            our_ask = our_bid + 1

        buy_cap = POSITION_LIMIT - position
        sell_cap = POSITION_LIMIT + position

        if buy_cap > 0:
            orders.append(Order(PRODUCT, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(PRODUCT, our_ask, -sell_cap))

        return orders

    def trade_tomatoes(self, state: TradingState) -> List[Order]:
        PRODUCT = "TOMATOES"
        POSITION_LIMIT = 80

        position = state.position.get(PRODUCT, 0)
        order_depth = state.order_depths[PRODUCT]
        orders: List[Order] = []

        # Compute Wall Mid
        bid_wall_price, ask_wall_price = None, None
        bid_wall_vol, ask_wall_vol = 0, 0

        for price, vol in order_depth.buy_orders.items():
            if vol > bid_wall_vol:
                bid_wall_vol = vol
                bid_wall_price = price

        for price, vol in order_depth.sell_orders.items():
            if abs(vol) > ask_wall_vol:
                ask_wall_vol = abs(vol)
                ask_wall_price = price

        if bid_wall_price is not None and ask_wall_price is not None:
            fair_value = (bid_wall_price + ask_wall_price) / 2
        else:
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
            if best_bid and best_ask:
                fair_value = (best_bid + best_ask) / 2
            else:
                return orders

        fv_int = round(fair_value)

        # Phase 1: TAKE — sweep at or better than fair value
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= fv_int:
                    ask_vol = order_depth.sell_orders[ask_price]
                    buy_qty = min(-ask_vol, POSITION_LIMIT - position)
                    if buy_qty > 0:
                        orders.append(Order(PRODUCT, ask_price, buy_qty))
                        position += buy_qty
                else:
                    break

        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price >= fv_int:
                    bid_vol = order_depth.buy_orders[bid_price]
                    sell_qty = min(bid_vol, POSITION_LIMIT + position)
                    if sell_qty > 0:
                        orders.append(Order(PRODUCT, bid_price, -sell_qty))
                        position -= sell_qty
                else:
                    break

        # Phase 2: MAKE — overbid/undercut existing book
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else fv_int - 7
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else fv_int + 7

        our_bid = min(best_bid + 1, fv_int - 1)
        our_ask = max(best_ask - 1, fv_int + 1)

        # Stronger skew for TOMATOES — it drifts more
        skew = round(-position * 3 / POSITION_LIMIT)
        our_bid += skew
        our_ask += skew

        our_bid = min(our_bid, fv_int - 1)
        our_ask = max(our_ask, fv_int + 1)
        if our_bid >= our_ask:
            our_ask = our_bid + 1

        buy_cap = POSITION_LIMIT - position
        sell_cap = POSITION_LIMIT + position

        if buy_cap > 0:
            orders.append(Order(PRODUCT, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(PRODUCT, our_ask, -sell_cap))

        return orders