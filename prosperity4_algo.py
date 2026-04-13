"""
IMC Prosperity 4 — Tutorial Round Algorithm v3
================================================
v3 fixes from v2:
  - EMERALDS: Stop taking at 10000 (was 74% of volume, 0 edge!)
    Only take strictly better than fair. Quote tighter MAKEs at 9998/10002.
  - TOMATOES: Tighten spread from wall+3 to wall+2, reduce take edge to 1
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState
)
import json
from typing import Any

EMERALDS_LIMIT = 50
TOMATOES_LIMIT = 50
EMERALDS_FAIR = 10_000
TOMATOES_MAX_POS = 20


class Trader:

    def __init__(self):
        self.tomatoes_ema = None

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
                self.tomatoes_ema = trader_data.get("tomatoes_ema")
            except Exception:
                pass

        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self.trade_emeralds(state)
        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self.trade_tomatoes(state)

        trader_data["tomatoes_ema"] = self.tomatoes_ema
        return result, conversions, json.dumps(trader_data)

    def trade_emeralds(self, state: TradingState) -> list[Order]:
        """
        v3 key change: DON'T take at 10000 — it's zero edge.
        Only take if strictly favorable (buy < 10000, sell > 10000).
        Quote MAKEs tighter at 9998/10002 to get more fills with real edge.
        """
        orders: list[Order] = []
        depth = state.order_depths["EMERALDS"]
        pos = state.position.get("EMERALDS", 0)
        fair = EMERALDS_FAIR
        limit = EMERALDS_LIMIT
        buy_cap = limit - pos
        sell_cap = limit + pos

        # PHASE 1: TAKE only strictly favorable crosses (NOT at fair!)
        if depth.sell_orders:
            for ask_price in sorted(depth.sell_orders.keys()):
                if ask_price < fair and buy_cap > 0:  # strictly less than 10000
                    vol = abs(depth.sell_orders[ask_price])
                    qty = min(vol, buy_cap)
                    orders.append(Order("EMERALDS", ask_price, qty))
                    buy_cap -= qty
                    pos += qty

        if depth.buy_orders:
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_price > fair and sell_cap > 0:  # strictly greater than 10000
                    vol = depth.buy_orders[bid_price]
                    qty = min(vol, sell_cap)
                    orders.append(Order("EMERALDS", bid_price, -qty))
                    sell_cap -= qty
                    pos -= qty

        # PHASE 2: MAKE at tight levels around fair
        # v3: quote at 9998/10002 (edge=2) as primary, 9996/10004 as backup
        # Inventory skew: shift quotes toward closing position
        inv_ratio = pos / limit  # -1 to +1
        skew = round(inv_ratio * 2)

        # Primary: tight quotes close to fair (more fills, lower edge)
        bid1 = fair - 2 + skew  # 9998 when flat
        ask1 = fair + 2 + skew  # 10002 when flat

        # Backup: wider quotes (fewer fills, higher edge)
        bid2 = fair - 4 + skew  # 9996 when flat
        ask2 = fair + 4 + skew  # 10004 when flat

        # Safety: never cross fair
        bid1 = min(bid1, fair - 1)
        bid2 = min(bid2, fair - 1)
        ask1 = max(ask1, fair + 1)
        ask2 = max(ask2, fair + 1)

        # Split volume between levels
        bid1_qty = min(buy_cap * 2 // 3, buy_cap)
        bid2_qty = buy_cap - bid1_qty
        ask1_qty = min(sell_cap * 2 // 3, sell_cap)
        ask2_qty = sell_cap - ask1_qty

        if bid1_qty > 0:
            orders.append(Order("EMERALDS", bid1, bid1_qty))
            buy_cap -= bid1_qty
        if bid2_qty > 0 and buy_cap > 0:
            orders.append(Order("EMERALDS", bid2, min(bid2_qty, buy_cap)))
        if ask1_qty > 0:
            orders.append(Order("EMERALDS", ask1, -ask1_qty))
            sell_cap -= ask1_qty
        if ask2_qty > 0 and sell_cap > 0:
            orders.append(Order("EMERALDS", ask2, -min(ask2_qty, sell_cap)))

        return orders

    def trade_tomatoes(self, state: TradingState) -> list[Order]:
        """
        v3: Tighten spread from wall+3 to wall+2, reduce take edge to 1.
        Keep position cap and aggressive flattening from v2.
        """
        orders: list[Order] = []
        depth = state.order_depths["TOMATOES"]
        pos = state.position.get("TOMATOES", 0)
        limit = TOMATOES_LIMIT
        buy_cap = limit - pos
        sell_cap = limit + pos

        wall_bid, wall_ask = self._find_walls(depth)
        if wall_bid is None or wall_ask is None:
            return orders

        wall_mid = (wall_bid + wall_ask) / 2
        if self.tomatoes_ema is None:
            self.tomatoes_ema = wall_mid
        else:
            self.tomatoes_ema = 0.3 * wall_mid + 0.7 * self.tomatoes_ema

        fair = round(wall_mid)

        # PHASE 1: TAKE with edge >= 1 (tighter than v2's edge >= 2)
        if depth.sell_orders:
            for ask_price in sorted(depth.sell_orders.keys()):
                if ask_price < fair - 1 and buy_cap > 0:
                    vol = abs(depth.sell_orders[ask_price])
                    qty = min(vol, buy_cap, TOMATOES_MAX_POS - max(pos, 0))
                    if qty > 0:
                        orders.append(Order("TOMATOES", ask_price, qty))
                        buy_cap -= qty
                        pos += qty

        if depth.buy_orders:
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_price > fair + 1 and sell_cap > 0:
                    vol = depth.buy_orders[bid_price]
                    qty = min(vol, sell_cap, TOMATOES_MAX_POS + min(pos, 0))
                    if qty > 0:
                        orders.append(Order("TOMATOES", bid_price, -qty))
                        sell_cap -= qty
                        pos -= qty

        # PHASE 2: FLATTEN inventory aggressively
        abs_pos = abs(pos)
        if abs_pos > 5:
            if pos > 0 and sell_cap > 0:
                flatten_qty = min(pos - 2, sell_cap)
                if flatten_qty > 0:
                    orders.append(Order("TOMATOES", fair, -flatten_qty))
                    sell_cap -= flatten_qty
                    pos -= flatten_qty
            elif pos < 0 and buy_cap > 0:
                flatten_qty = min(-pos - 2, buy_cap)
                if flatten_qty > 0:
                    orders.append(Order("TOMATOES", fair, flatten_qty))
                    buy_cap -= flatten_qty
                    pos += flatten_qty

        # PHASE 3: MAKE with tighter spread (wall+2 instead of wall+3)
        inv_ratio = abs(pos) / limit
        extra_spread = round(inv_ratio * 6)  # widen more aggressively with inventory

        base_offset = 2  # tighter than v2's 3
        bid_offset = base_offset + extra_spread
        ask_offset = base_offset + extra_spread

        inv_skew = round(pos / limit * 3)
        bid_price = wall_bid + bid_offset - inv_skew
        ask_price = wall_ask - ask_offset - inv_skew

        bid_price = min(bid_price, fair - 1)
        ask_price = max(ask_price, fair + 1)
        if bid_price >= ask_price:
            bid_price = fair - 2
            ask_price = fair + 2

        max_new_buy = max(0, TOMATOES_MAX_POS - max(pos, 0))
        max_new_sell = max(0, TOMATOES_MAX_POS + min(pos, 0))
        bid_qty = min(buy_cap, max_new_buy)
        ask_qty = min(sell_cap, max_new_sell)

        if bid_qty > 0:
            orders.append(Order("TOMATOES", bid_price, bid_qty))
        if ask_qty > 0:
            orders.append(Order("TOMATOES", ask_price, -ask_qty))

        return orders

    def _find_walls(self, depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None, None
        wall_bid = max(depth.buy_orders.keys(), key=lambda p: depth.buy_orders[p])
        max_bv = depth.buy_orders[wall_bid]
        candidates = [p for p in depth.buy_orders if depth.buy_orders[p] == max_bv]
        wall_bid = max(candidates)
        wall_ask = min(depth.sell_orders.keys(), key=lambda p: depth.sell_orders[p])
        max_av = depth.sell_orders[wall_ask]
        candidates = [p for p in depth.sell_orders if depth.sell_orders[p] == max_av]
        wall_ask = min(candidates)
        return wall_bid, wall_ask