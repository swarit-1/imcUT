"""
IMC Prosperity 4 — Tutorial Round Algorithm v2
================================================
FIXES from v1:
  1. TOMATOES: Much wider spread, capped position, aggressive inventory decay
  2. EMERALDS: Multi-level quoting, aggressive taking at 10000
  3. Both: Position-proportional spread widening to prevent runaway inventory
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState
)
import json
import math
from typing import Any

# ─── POSITION LIMITS (check wiki and adjust!) ───────────
EMERALDS_LIMIT = 50
TOMATOES_LIMIT = 50

# ─── EMERALDS CONFIG ────────────────────────────────────
EMERALDS_FAIR = 10_000

# ─── TOMATOES CONFIG ────────────────────────────────────
# Key insight: edge is ~7/side but holding risk grows fast
# Be MUCH more conservative than v1
TOMATOES_MAX_POS = 20        # soft cap: don't hold more than this
TOMATOES_BASE_SPREAD = 3     # base offset inside walls (wider than v1's 1)
TOMATOES_INV_PENALTY = 1.5   # extra spread per unit of inventory ratio


class Trader:

    def __init__(self):
        self.tomatoes_ema = None

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        # Restore state
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

    # ═══════════════════════════════════════════════════════
    #  EMERALDS — Aggressive Fixed-Value Market Making
    # ═══════════════════════════════════════════════════════
    def trade_emeralds(self, state: TradingState) -> list[Order]:
        orders: list[Order] = []
        depth = state.order_depths["EMERALDS"]
        pos = state.position.get("EMERALDS", 0)
        fair = EMERALDS_FAIR
        limit = EMERALDS_LIMIT

        buy_cap = limit - pos
        sell_cap = limit + pos

        # ── PHASE 1: TAKE all crosses ────────────────────
        # Buy anything offered at or below fair value
        if depth.sell_orders:
            for ask_price in sorted(depth.sell_orders.keys()):
                if ask_price <= fair and buy_cap > 0:
                    vol = abs(depth.sell_orders[ask_price])
                    qty = min(vol, buy_cap)
                    orders.append(Order("EMERALDS", ask_price, qty))
                    buy_cap -= qty
                    pos += qty

        # Sell anything bid at or above fair value
        if depth.buy_orders:
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_price >= fair and sell_cap > 0:
                    vol = depth.buy_orders[bid_price]
                    qty = min(vol, sell_cap)
                    orders.append(Order("EMERALDS", bid_price, -qty))
                    sell_cap -= qty
                    pos -= qty

        # ── PHASE 2: MAKE at multiple levels ─────────────
        # Inventory-aware spread: shift quotes toward flattening
        # With 0 inventory: bid 9997, 9996, 9995 / ask 10003, 10004, 10005
        # Skew everything by inventory
        inv_ratio = pos / limit  # -1 to +1

        # Primary quotes: tight spread, captures most post-us takers
        # Secondary quotes: wider, catches bigger takers
        bid_levels = [
            (fair - 3 + round(inv_ratio * 2), buy_cap // 2),   # aggressive
            (fair - 5 + round(inv_ratio * 2), buy_cap - buy_cap // 2),  # backup
        ]
        ask_levels = [
            (fair + 3 + round(inv_ratio * 2), sell_cap // 2),   # aggressive
            (fair + 5 + round(inv_ratio * 2), sell_cap - sell_cap // 2),  # backup
        ]

        for price, qty in bid_levels:
            price = min(price, fair - 1)  # never bid at fair or above
            if qty > 0 and buy_cap > 0:
                actual_qty = min(qty, buy_cap)
                orders.append(Order("EMERALDS", price, actual_qty))
                buy_cap -= actual_qty

        for price, qty in ask_levels:
            price = max(price, fair + 1)  # never ask at fair or below
            if qty > 0 and sell_cap > 0:
                actual_qty = min(qty, sell_cap)
                orders.append(Order("EMERALDS", price, -actual_qty))
                sell_cap -= actual_qty

        return orders

    # ═══════════════════════════════════════════════════════
    #  TOMATOES — Conservative Wall Mid Market Making
    # ═══════════════════════════════════════════════════════
    def trade_tomatoes(self, state: TradingState) -> list[Order]:
        orders: list[Order] = []
        depth = state.order_depths["TOMATOES"]
        pos = state.position.get("TOMATOES", 0)
        limit = TOMATOES_LIMIT

        buy_cap = limit - pos
        sell_cap = limit + pos

        # ── Compute Wall Mid ─────────────────────────────
        wall_bid, wall_ask = self._find_walls(depth)
        if wall_bid is None or wall_ask is None:
            return orders

        wall_mid = (wall_bid + wall_ask) / 2

        # Update EMA (smoothed fair value)
        if self.tomatoes_ema is None:
            self.tomatoes_ema = wall_mid
        else:
            self.tomatoes_ema = 0.3 * wall_mid + 0.7 * self.tomatoes_ema

        fair = round(wall_mid)  # use instantaneous wall mid, not EMA

        # ── PHASE 1: TAKE only high-edge opportunities ───
        # Only take if edge >= 2 (more conservative than v1)
        min_take_edge = 2

        if depth.sell_orders:
            for ask_price in sorted(depth.sell_orders.keys()):
                if ask_price < fair - min_take_edge and buy_cap > 0:
                    vol = abs(depth.sell_orders[ask_price])
                    qty = min(vol, buy_cap, TOMATOES_MAX_POS - max(pos, 0))
                    if qty > 0:
                        orders.append(Order("TOMATOES", ask_price, qty))
                        buy_cap -= qty
                        pos += qty

        if depth.buy_orders:
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_price > fair + min_take_edge and sell_cap > 0:
                    vol = depth.buy_orders[bid_price]
                    qty = min(vol, sell_cap, TOMATOES_MAX_POS + min(pos, 0))
                    if qty > 0:
                        orders.append(Order("TOMATOES", bid_price, -qty))
                        sell_cap -= qty
                        pos -= qty

        # ── PHASE 2: FLATTEN inventory aggressively ──────
        # Key v2 change: if we hold ANY significant position, start reducing
        # at fair price to prevent holding-period risk from eroding edge
        abs_pos = abs(pos)

        if abs_pos > 5:
            # Aggressively flatten at fair or slightly better
            if pos > 0 and sell_cap > 0:
                # We're long, post asks at fair to dump
                flatten_qty = min(pos - 2, sell_cap)
                if flatten_qty > 0:
                    orders.append(Order("TOMATOES", fair, -flatten_qty))
                    sell_cap -= flatten_qty
                    pos -= flatten_qty
            elif pos < 0 and buy_cap > 0:
                # We're short, post bids at fair to cover
                flatten_qty = min(-pos - 2, buy_cap)
                if flatten_qty > 0:
                    orders.append(Order("TOMATOES", fair, flatten_qty))
                    buy_cap -= flatten_qty
                    pos += flatten_qty

        # ── PHASE 3: MAKE with wider spread, capped size ─
        # Inventory-proportional spread widening
        inv_ratio = abs(pos) / limit
        extra_spread = round(inv_ratio * TOMATOES_INV_PENALTY * 4)

        # Dynamic spread: wider when we have inventory
        bid_offset = TOMATOES_BASE_SPREAD + extra_spread
        ask_offset = TOMATOES_BASE_SPREAD + extra_spread

        # Skew toward reducing inventory
        inv_skew = round(pos / limit * 3)  # positive when long → lower prices
        bid_price = wall_bid + bid_offset - inv_skew
        ask_price = wall_ask - ask_offset - inv_skew

        # Safety rails
        bid_price = min(bid_price, fair - 1)
        ask_price = max(ask_price, fair + 1)
        if bid_price >= ask_price:
            bid_price = fair - 2
            ask_price = fair + 2

        # Cap how much new inventory we take on
        max_new_buy = max(0, TOMATOES_MAX_POS - max(pos, 0))
        max_new_sell = max(0, TOMATOES_MAX_POS + min(pos, 0))

        bid_qty = min(buy_cap, max_new_buy)
        ask_qty = min(sell_cap, max_new_sell)

        if bid_qty > 0:
            orders.append(Order("TOMATOES", bid_price, bid_qty))
        if ask_qty > 0:
            orders.append(Order("TOMATOES", ask_price, -ask_qty))

        return orders

    # ═══════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════
    def _find_walls(self, depth: OrderDepth):
        """Find wall (deepest liquidity) bid and ask levels."""
        if not depth.buy_orders or not depth.sell_orders:
            return None, None

        # Wall bid: price level with highest volume
        wall_bid = max(depth.buy_orders.keys(),
                       key=lambda p: depth.buy_orders[p])
        max_bv = depth.buy_orders[wall_bid]
        candidates = [p for p in depth.buy_orders if depth.buy_orders[p] == max_bv]
        wall_bid = max(candidates)  # highest price with max volume

        # Wall ask: price level with most volume (most negative = deepest)
        wall_ask = min(depth.sell_orders.keys(),
                       key=lambda p: depth.sell_orders[p])
        max_av = depth.sell_orders[wall_ask]
        candidates = [p for p in depth.sell_orders if depth.sell_orders[p] == max_av]
        wall_ask = min(candidates)  # lowest price with max volume

        return wall_bid, wall_ask