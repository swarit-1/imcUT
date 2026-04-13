"""
IMC Prosperity 4 — Tutorial Round Algorithm v3
================================================
Changes from v2 (data-driven):
  1. EMERALDS: microprice/imbalance skew (corr +0.62 vs next mid return),
     wider base spread (5 ticks) for better edge per fill, asymmetric
     quoting based on imbalance.
  2. TOMATOES: fair = EMA (was dead code in v2), microprice blend,
     L1 imbalance skew, properly normalized inventory skew.
  3. Cleanup: removed dead code in _find_walls, fixed inv_ratio
     normalization to use MAX_POS not LIMIT.
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState
)
import json
import math
from typing import Any

# ─── POSITION LIMITS ────────────────────────────────────
EMERALDS_LIMIT = 50
TOMATOES_LIMIT = 50

# ─── EMERALDS CONFIG ────────────────────────────────────
EMERALDS_FAIR = 10_000
EMERALDS_BASE_OFFSET = 4      # post at fair±4 (was ±3) for better edge
EMERALDS_BACKUP_OFFSET = 6
EMERALDS_IMB_SKEW = 2         # max ticks to shift quotes from imbalance

# ─── TOMATOES CONFIG ────────────────────────────────────
TOMATOES_MAX_POS = 25
TOMATOES_BASE_SPREAD = 3
TOMATOES_INV_PENALTY = 1.5
TOMATOES_EMA_ALPHA = 0.2      # smoothing for fair value
TOMATOES_IMB_SKEW = 2


def microprice(depth: OrderDepth) -> float | None:
    """Volume-weighted price using L1 — predictive of next mid move."""
    if not depth.buy_orders or not depth.sell_orders:
        return None
    best_bid = max(depth.buy_orders.keys())
    best_ask = min(depth.sell_orders.keys())
    bv = depth.buy_orders[best_bid]
    av = abs(depth.sell_orders[best_ask])
    if bv + av == 0:
        return (best_bid + best_ask) / 2
    return (best_bid * av + best_ask * bv) / (bv + av)


def l1_imbalance(depth: OrderDepth) -> float:
    """Returns value in [-1, +1]. Positive = bid heavy → price tends up."""
    if not depth.buy_orders or not depth.sell_orders:
        return 0.0
    best_bid = max(depth.buy_orders.keys())
    best_ask = min(depth.sell_orders.keys())
    bv = depth.buy_orders[best_bid]
    av = abs(depth.sell_orders[best_ask])
    if bv + av == 0:
        return 0.0
    return (bv - av) / (bv + av)


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

    # ═══════════════════════════════════════════════════════
    #  EMERALDS — Imbalance-Aware Fixed-Value Market Making
    # ═══════════════════════════════════════════════════════
    def trade_emeralds(self, state: TradingState) -> list[Order]:
        orders: list[Order] = []
        depth = state.order_depths["EMERALDS"]
        pos = state.position.get("EMERALDS", 0)
        fair = EMERALDS_FAIR
        limit = EMERALDS_LIMIT

        buy_cap = limit - pos
        sell_cap = limit + pos

        imb = l1_imbalance(depth)  # [-1, +1]

        # ── PHASE 1: TAKE crosses against true fair ──────
        if depth.sell_orders:
            for ask_price in sorted(depth.sell_orders.keys()):
                # Take aggressively when ask < fair, OR when ask == fair
                # and book leans bid-heavy (price about to rise)
                threshold = fair if imb > 0.3 else fair - 1
                if ask_price <= threshold and buy_cap > 0:
                    vol = abs(depth.sell_orders[ask_price])
                    qty = min(vol, buy_cap)
                    orders.append(Order("EMERALDS", ask_price, qty))
                    buy_cap -= qty
                    pos += qty

        if depth.buy_orders:
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                threshold = fair if imb < -0.3 else fair + 1
                if bid_price >= threshold and sell_cap > 0:
                    vol = depth.buy_orders[bid_price]
                    qty = min(vol, sell_cap)
                    orders.append(Order("EMERALDS", bid_price, -qty))
                    sell_cap -= qty
                    pos -= qty

        # ── PHASE 2: MAKE with imbalance + inventory skew ─
        # Inventory skew: shift quotes to flatten
        inv_skew = round(pos / limit * 2)  # max ±2

        # Imbalance skew: bid heavier book → expect price up → raise both quotes
        imb_skew = round(imb * EMERALDS_IMB_SKEW)

        bid_price = fair - EMERALDS_BASE_OFFSET - inv_skew + imb_skew
        ask_price = fair + EMERALDS_BASE_OFFSET - inv_skew + imb_skew

        # Asymmetric size: when bid-heavy, post smaller asks (don't want to sell into rally)
        if imb > 0.3:
            bid_size = buy_cap
            ask_size = max(1, sell_cap // 2)
        elif imb < -0.3:
            bid_size = max(1, buy_cap // 2)
            ask_size = sell_cap
        else:
            bid_size = buy_cap
            ask_size = sell_cap

        # Safety rails
        bid_price = min(bid_price, fair - 1)
        ask_price = max(ask_price, fair + 1)

        if bid_size > 0 and buy_cap > 0:
            qty = min(bid_size, buy_cap)
            orders.append(Order("EMERALDS", bid_price, qty))
            buy_cap -= qty

        if ask_size > 0 and sell_cap > 0:
            qty = min(ask_size, sell_cap)
            orders.append(Order("EMERALDS", ask_price, -qty))
            sell_cap -= qty

        # Backup deeper levels using leftover capacity
        if buy_cap > 0:
            orders.append(Order("EMERALDS", fair - EMERALDS_BACKUP_OFFSET - inv_skew, buy_cap))
        if sell_cap > 0:
            orders.append(Order("EMERALDS", fair + EMERALDS_BACKUP_OFFSET - inv_skew, -sell_cap))

        return orders

    # ═══════════════════════════════════════════════════════
    #  TOMATOES — Drift-Aware MM with Microprice Fair
    # ═══════════════════════════════════════════════════════
    def trade_tomatoes(self, state: TradingState) -> list[Order]:
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
        mp = microprice(depth)
        imb = l1_imbalance(depth)

        # Blend wall_mid (anchored, slow) with microprice (responsive)
        instant_fair = 0.5 * wall_mid + 0.5 * (mp if mp is not None else wall_mid)

        # Update EMA for trend tracking across drift
        if self.tomatoes_ema is None:
            self.tomatoes_ema = instant_fair
        else:
            self.tomatoes_ema = (TOMATOES_EMA_ALPHA * instant_fair
                                 + (1 - TOMATOES_EMA_ALPHA) * self.tomatoes_ema)

        # Use EMA as fair (FIX: v2 used wall_mid, EMA was dead code)
        fair = round(self.tomatoes_ema)

        # ── PHASE 1: TAKE high-edge crosses ──────────────
        # Lower edge requirement when imbalance confirms direction
        for ask_price in sorted(depth.sell_orders.keys()):
            edge = fair - ask_price
            min_edge = 1 if imb > 0.3 else 2
            if edge >= min_edge and buy_cap > 0:
                vol = abs(depth.sell_orders[ask_price])
                qty = min(vol, buy_cap, max(0, TOMATOES_MAX_POS - pos))
                if qty > 0:
                    orders.append(Order("TOMATOES", ask_price, qty))
                    buy_cap -= qty
                    pos += qty

        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            edge = bid_price - fair
            min_edge = 1 if imb < -0.3 else 2
            if edge >= min_edge and sell_cap > 0:
                vol = depth.buy_orders[bid_price]
                qty = min(vol, sell_cap, max(0, TOMATOES_MAX_POS + pos))
                if qty > 0:
                    orders.append(Order("TOMATOES", bid_price, -qty))
                    sell_cap -= qty
                    pos -= qty

        # ── PHASE 2: FLATTEN inventory ───────────────────
        if abs(pos) > 5:
            if pos > 0 and sell_cap > 0:
                # Post one tick above fair — still inside wall, recovers spread
                flatten_px = fair + 1
                flatten_qty = min(pos - 2, sell_cap)
                if flatten_qty > 0:
                    orders.append(Order("TOMATOES", flatten_px, -flatten_qty))
                    sell_cap -= flatten_qty
                    pos -= flatten_qty
            elif pos < 0 and buy_cap > 0:
                flatten_px = fair - 1
                flatten_qty = min(-pos - 2, buy_cap)
                if flatten_qty > 0:
                    orders.append(Order("TOMATOES", flatten_px, flatten_qty))
                    buy_cap -= flatten_qty
                    pos += flatten_qty

        # ── PHASE 3: MAKE with skews ─────────────────────
        # FIX: normalize against MAX_POS, not LIMIT (v2 muted the skew)
        inv_ratio = pos / TOMATOES_MAX_POS  # [-1, +1]
        extra_spread = round(abs(inv_ratio) * TOMATOES_INV_PENALTY * 2)

        bid_offset = TOMATOES_BASE_SPREAD + extra_spread
        ask_offset = TOMATOES_BASE_SPREAD + extra_spread

        inv_skew = round(inv_ratio * 2)
        imb_skew = round(imb * TOMATOES_IMB_SKEW)

        bid_price = wall_bid + bid_offset - inv_skew + imb_skew
        ask_price = wall_ask - ask_offset - inv_skew + imb_skew

        # Safety: keep at least 1 tick from fair, never crossed
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

    # ═══════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════
    def _find_walls(self, depth: OrderDepth):
        """Find wall (deepest liquidity) bid and ask levels."""
        if not depth.buy_orders or not depth.sell_orders:
            return None, None

        max_bv = max(depth.buy_orders.values())
        wall_bid = max(p for p, v in depth.buy_orders.items() if v == max_bv)

        max_av = min(depth.sell_orders.values())  # most negative
        wall_ask = min(p for p, v in depth.sell_orders.items() if v == max_av)

        return wall_bid, wall_ask
