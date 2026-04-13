"""
IMC Prosperity 4 — Tutorial Round Algorithm
============================================
Products: EMERALDS (fixed fair value ~10,000) and TOMATOES (slow random walk)

Strategy based on insights from Prosperity 3 top finishers:
  - Frankfurt Hedgehogs (2nd globally)
  - CMU Physics (7th globally, 1st USA)
  - camel_case / Jasper (25th globally, 1st Netherlands)

Key principles applied:
  1. Deep structural understanding before strategy design
  2. Simple, robust strategies with minimal parameters
  3. Wall Mid as fair value estimator
  4. Overbid/undercut to capture taker flow
  5. Inventory management to maintain risk capacity
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState
)
import json
import math
from typing import Any

# ─── CONFIGURATION ──────────────────────────────────────────
EMERALDS_FAIR = 10_000
EMERALDS_LIMIT = 50       # position limit (adjust per wiki)
TOMATOES_LIMIT = 50       # position limit (adjust per wiki)

# EMERALDS: how far inside the wall to quote
EMERALDS_MAKE_EDGE = 7    # quote at 9993 / 10007
EMERALDS_TAKE_EDGE = 0    # take anything at fair or better

# TOMATOES: quoting parameters
TOMATOES_MAKE_OFFSET = 1  # quote 1 inside the wall (wall_bid+1 / wall_ask-1)
TOMATOES_TAKE_EDGE = 1    # take anything with >= 1 edge vs wall mid
TOMATOES_INV_SKEW = 2     # shift quotes by this per unit of inventory


class Trader:
    """Main trader class submitted to the Prosperity platform."""

    def __init__(self):
        self.tomatoes_ema = None
        self.ema_alpha = 0.2  # fast EMA for tomatoes wall mid

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

        # ─── EMERALDS ───────────────────────────────────────
        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self.trade_emeralds(state)

        # ─── TOMATOES ───────────────────────────────────────
        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self.trade_tomatoes(state)

        # Save state
        trader_data["tomatoes_ema"] = self.tomatoes_ema
        trader_data_str = json.dumps(trader_data)

        return result, conversions, trader_data_str

    # ═══════════════════════════════════════════════════════
    #  EMERALDS — Fixed Fair Value Market Making
    # ═══════════════════════════════════════════════════════
    def trade_emeralds(self, state: TradingState) -> list[Order]:
        """
        EMERALDS has a fixed true price at 10,000.
        
        Strategy (from Frankfurt Hedgehogs' Rainforest Resin approach):
        1. TAKE: immediately buy anything offered below fair,
                 immediately sell anything bid above fair.
        2. MAKE: place passive bids at 9993 and asks at 10007
                 (overbidding the wall at 9992, undercutting wall at 10008).
        3. INVENTORY: skew quotes to manage position toward zero.
                 If heavily long, lower ask closer to 10000 to sell.
                 If heavily short, raise bid closer to 10000 to buy.
        """
        orders: list[Order] = []
        depth = state.order_depths["EMERALDS"]
        pos = state.position.get("EMERALDS", 0)
        fair = EMERALDS_FAIR
        limit = EMERALDS_LIMIT

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # ── Phase 1: TAKE profitable orders ──────────────
        # Buy from anyone selling below fair
        if depth.sell_orders:
            for ask_price in sorted(depth.sell_orders.keys()):
                if ask_price < fair and buy_capacity > 0:
                    ask_vol = abs(depth.sell_orders[ask_price])
                    take_qty = min(ask_vol, buy_capacity)
                    orders.append(Order("EMERALDS", ask_price, take_qty))
                    buy_capacity -= take_qty
                    pos += take_qty

        # Sell to anyone buying above fair
        if depth.buy_orders:
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_price > fair and sell_capacity > 0:
                    bid_vol = depth.buy_orders[bid_price]
                    take_qty = min(bid_vol, sell_capacity)
                    orders.append(Order("EMERALDS", bid_price, -take_qty))
                    sell_capacity -= take_qty
                    pos -= take_qty

        # ── Phase 2: Flatten at fair if inventory is extreme ─
        # If |pos| > 70% of limit, trade at fair to free capacity
        inv_threshold = int(limit * 0.7)
        if pos > inv_threshold and sell_capacity > 0:
            flatten_qty = min(pos - inv_threshold // 2, sell_capacity)
            if flatten_qty > 0:
                orders.append(Order("EMERALDS", fair, -flatten_qty))
                sell_capacity -= flatten_qty
                pos -= flatten_qty

        if pos < -inv_threshold and buy_capacity > 0:
            flatten_qty = min(-pos - inv_threshold // 2, buy_capacity)
            if flatten_qty > 0:
                orders.append(Order("EMERALDS", fair, flatten_qty))
                buy_capacity -= flatten_qty
                pos += flatten_qty

        # ── Phase 3: MAKE passive orders ─────────────────
        # Quote inside the wall, skewed by inventory
        # Inventory skew: shift both quotes toward closing inventory
        inv_skew = round(pos / limit * 2)  # -2 to +2 range

        # Bid: overbid the wall (9992) → 9993, adjusted by skew
        bid_price = fair - EMERALDS_MAKE_EDGE + inv_skew
        bid_price = min(bid_price, fair - 1)  # never bid at or above fair

        # Ask: undercut the wall (10008) → 10007, adjusted by skew
        ask_price = fair + EMERALDS_MAKE_EDGE + inv_skew
        ask_price = max(ask_price, fair + 1)  # never ask at or below fair

        if buy_capacity > 0:
            orders.append(Order("EMERALDS", bid_price, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order("EMERALDS", ask_price, -sell_capacity))

        return orders

    # ═══════════════════════════════════════════════════════
    #  TOMATOES — Wall Mid Market Making
    # ═══════════════════════════════════════════════════════
    def trade_tomatoes(self, state: TradingState) -> list[Order]:
        """
        TOMATOES follows a slow random walk (like Kelp in Prosperity 3).
        
        Strategy (from Frankfurt Hedgehogs / CMU Physics):
        1. Estimate true price via Wall Mid (average of wall bid & wall ask).
        2. TAKE: buy below wall_mid, sell above wall_mid.
        3. MAKE: quote 1 tick inside the walls, skewed by inventory.
        4. Mean reversion is strong (autocorr ≈ -0.43) — use EMA to detect
           short-term overshoots and lean into reversion.
        """
        orders: list[Order] = []
        depth = state.order_depths["TOMATOES"]
        pos = state.position.get("TOMATOES", 0)
        limit = TOMATOES_LIMIT

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # ── Compute Wall Mid ─────────────────────────────
        wall_mid = self._compute_wall_mid(depth)
        if wall_mid is None:
            return orders  # can't trade without fair price estimate

        # Update EMA
        if self.tomatoes_ema is None:
            self.tomatoes_ema = wall_mid
        else:
            self.tomatoes_ema = self.ema_alpha * wall_mid + (1 - self.ema_alpha) * self.tomatoes_ema

        fair = wall_mid

        # ── Phase 1: TAKE profitable orders ──────────────
        if depth.sell_orders:
            for ask_price in sorted(depth.sell_orders.keys()):
                if ask_price < fair - TOMATOES_TAKE_EDGE and buy_capacity > 0:
                    ask_vol = abs(depth.sell_orders[ask_price])
                    take_qty = min(ask_vol, buy_capacity)
                    orders.append(Order("TOMATOES", ask_price, take_qty))
                    buy_capacity -= take_qty
                    pos += take_qty

        if depth.buy_orders:
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_price > fair + TOMATOES_TAKE_EDGE and sell_capacity > 0:
                    bid_vol = depth.buy_orders[bid_price]
                    take_qty = min(bid_vol, sell_capacity)
                    orders.append(Order("TOMATOES", bid_price, -take_qty))
                    sell_capacity -= take_qty
                    pos -= take_qty

        # ── Phase 2: Flatten heavy inventory at fair ─────
        inv_threshold = int(limit * 0.6)
        if pos > inv_threshold and sell_capacity > 0:
            flatten_qty = min(pos - inv_threshold // 2, sell_capacity)
            if flatten_qty > 0:
                orders.append(Order("TOMATOES", round(fair), -flatten_qty))
                sell_capacity -= flatten_qty
                pos -= flatten_qty

        if pos < -inv_threshold and buy_capacity > 0:
            flatten_qty = min(-pos - inv_threshold // 2, buy_capacity)
            if flatten_qty > 0:
                orders.append(Order("TOMATOES", round(fair), flatten_qty))
                buy_capacity -= flatten_qty
                pos += flatten_qty

        # ── Phase 3: MAKE passive orders ─────────────────
        # Find walls (deepest liquidity levels)
        wall_bid, wall_ask = self._find_walls(depth)
        if wall_bid is None or wall_ask is None:
            return orders

        # Inventory skew: shift quotes to reduce position
        inv_skew = round(pos / limit * TOMATOES_INV_SKEW)

        # Overbid the wall bid, undercut the wall ask
        bid_price = wall_bid + TOMATOES_MAKE_OFFSET + inv_skew
        ask_price = wall_ask - TOMATOES_MAKE_OFFSET + inv_skew

        # Safety: never cross fair
        bid_price = min(bid_price, round(fair) - 1)
        ask_price = max(ask_price, round(fair) + 1)

        # Ensure bid < ask
        if bid_price >= ask_price:
            bid_price = round(fair) - 1
            ask_price = round(fair) + 1

        if buy_capacity > 0:
            orders.append(Order("TOMATOES", bid_price, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order("TOMATOES", ask_price, -sell_capacity))

        return orders

    # ═══════════════════════════════════════════════════════
    #  Helper Methods
    # ═══════════════════════════════════════════════════════
    def _compute_wall_mid(self, depth: OrderDepth) -> float | None:
        """
        Compute Wall Mid: average of the deepest-liquidity bid and ask levels.
        
        The 'wall' is the price level with the most volume on each side.
        These walls are posted by designated market makers who know the true
        price and quote rounded values around it.
        
        This is more robust than raw mid price, which can be distorted by
        small overbids/undercuts from other participants.
        """
        if not depth.buy_orders or not depth.sell_orders:
            return None

        # Find wall bid (level with deepest volume)
        wall_bid = max(depth.buy_orders.keys(),
                       key=lambda p: depth.buy_orders[p])
        # Among equal volumes, pick the one closest to mid
        max_bid_vol = depth.buy_orders[wall_bid]
        wall_bid_candidates = [p for p in depth.buy_orders
                               if depth.buy_orders[p] == max_bid_vol]
        wall_bid = max(wall_bid_candidates)  # highest price with max vol

        # Find wall ask (level with deepest volume, negative)
        wall_ask = min(depth.sell_orders.keys(),
                       key=lambda p: depth.sell_orders[p])  # most negative = deepest
        max_ask_vol = depth.sell_orders[wall_ask]
        wall_ask_candidates = [p for p in depth.sell_orders
                               if depth.sell_orders[p] == max_ask_vol]
        wall_ask = min(wall_ask_candidates)  # lowest price with max vol

        return (wall_bid + wall_ask) / 2

    def _find_walls(self, depth: OrderDepth) -> tuple:
        """Return the wall bid and wall ask price levels."""
        if not depth.buy_orders or not depth.sell_orders:
            return None, None

        # Wall = deepest liquidity level
        wall_bid = max(depth.buy_orders.keys(),
                       key=lambda p: depth.buy_orders[p])
        max_bid_vol = depth.buy_orders[wall_bid]
        wall_bid_candidates = [p for p in depth.buy_orders
                               if depth.buy_orders[p] == max_bid_vol]
        wall_bid = max(wall_bid_candidates)

        wall_ask = min(depth.sell_orders.keys(),
                       key=lambda p: depth.sell_orders[p])
        max_ask_vol = depth.sell_orders[wall_ask]
        wall_ask_candidates = [p for p in depth.sell_orders
                               if depth.sell_orders[p] == max_ask_vol]
        wall_ask = min(wall_ask_candidates)

        return wall_bid, wall_ask
