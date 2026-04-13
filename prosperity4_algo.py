"""
IMC Prosperity 4 — Tutorial Round Algorithm v4
================================================
Changes from v3 (drawdown analysis from live run):
  1. Lowered imbalance thresholds (0.3 → 0.15/0.20) — L1 vols are usually
     similar so v3 thresholds rarely activated.
  2. Volatility regime tracking: rolling std of wall_mid widens spread and
     shrinks MAX_POS during fast moves (the 85K-99K drawdown).
  3. Drift detection on TOMATOES: when wall_mid diverges from EMA, bias
     hard in EMA direction — follow regime changes faster.
  4. Aggressive small-order taking: take inside-the-wall orders (vol < 5)
     at edge=1 unconditionally — these are uninformed probes.
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
EMERALDS_BACKUP_OFFSET = 6    # deeper safety-net level (still fixed)
EMERALDS_IMB_SKEW = 2
EMERALDS_IMB_THRESH = 0.15    # was 0.3 — most live signals were below 0.3

# ─── TOMATOES CONFIG ────────────────────────────────────
TOMATOES_MAX_POS = 25
TOMATOES_BASE_SPREAD = 3
TOMATOES_INV_PENALTY = 1.5
TOMATOES_EMA_ALPHA = 0.2
TOMATOES_IMB_SKEW = 2
TOMATOES_IMB_THRESH = 0.20    # was 0.3
TOMATOES_VOL_WINDOW = 20      # snapshots for rolling std
TOMATOES_VOL_HIGH = 1.5       # wall_mid std above this = "fast" regime
TOMATOES_DRIFT_THRESH = 3.0   # wall_mid - EMA gap that triggers hard bias
SMALL_ORDER_VOL = 5           # inside-wall probes ≤ this size are takeable @ edge≥1


def microprice(depth: OrderDepth) -> float | None:
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
        self.tomatoes_mid_history: list[float] = []  # for rolling vol

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
                self.tomatoes_ema = trader_data.get("tomatoes_ema")
                self.tomatoes_mid_history = trader_data.get("tomatoes_mid_history", [])
            except Exception:
                pass

        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self.trade_emeralds(state)

        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self.trade_tomatoes(state)

        trader_data["tomatoes_ema"] = self.tomatoes_ema
        trader_data["tomatoes_mid_history"] = self.tomatoes_mid_history[-TOMATOES_VOL_WINDOW:]
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

        imb = l1_imbalance(depth)
        thresh = EMERALDS_IMB_THRESH

        # Snapshot L1 BEFORE Phase 1 taking — used to anchor make quotes.
        # (depth dicts aren't mutated by appending Order objects, but the
        # *intent* is that anything we just took is no longer available.)
        l1_best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        l1_best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

        # ── PHASE 1: TAKE ────────────────────────────────
        if depth.sell_orders:
            for ask_price in sorted(depth.sell_orders.keys()):
                limit_px = fair if imb > thresh else fair - 1
                if ask_price <= limit_px and buy_cap > 0:
                    vol = abs(depth.sell_orders[ask_price])
                    qty = min(vol, buy_cap)
                    orders.append(Order("EMERALDS", ask_price, qty))
                    buy_cap -= qty
                    pos += qty

        if depth.buy_orders:
            for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
                limit_px = fair if imb < -thresh else fair + 1
                if bid_price >= limit_px and sell_cap > 0:
                    vol = depth.buy_orders[bid_price]
                    qty = min(vol, sell_cap)
                    orders.append(Order("EMERALDS", bid_price, -qty))
                    sell_cap -= qty
                    pos -= qty

        # ── PHASE 2: MAKE — penny the L1 book ────────────
        # Anchor to live book instead of a fixed offset from fair, so we
        # adapt when the L1 small orders move (e.g. 10000-priced inside-wall
        # orders that briefly compress the spread).
        inv_skew = round(pos / limit * 2)
        imb_skew = round(imb * EMERALDS_IMB_SKEW)

        # Penny: one tick better than the existing best on each side.
        # Fallbacks if the side is empty: use wall-style default (fair ± 8).
        base_bid = (l1_best_bid + 1) if l1_best_bid is not None else (fair - 8)
        base_ask = (l1_best_ask - 1) if l1_best_ask is not None else (fair + 8)

        bid_price = base_bid - inv_skew + imb_skew
        ask_price = base_ask - inv_skew + imb_skew

        # Safety rails:
        #   1) never quote at or through fair (keep ≥1 tick away)
        #   2) never cross our own bid/ask
        bid_price = min(bid_price, fair - 1)
        ask_price = max(ask_price, fair + 1)
        if bid_price >= ask_price:
            bid_price = fair - 1
            ask_price = fair + 1

        # Asymmetric size based on imbalance — same as v4 logic
        if imb > thresh:
            bid_size = buy_cap
            ask_size = max(1, sell_cap // 2)
        elif imb < -thresh:
            bid_size = max(1, buy_cap // 2)
            ask_size = sell_cap
        else:
            bid_size = buy_cap
            ask_size = sell_cap

        if bid_size > 0 and buy_cap > 0:
            qty = min(bid_size, buy_cap)
            orders.append(Order("EMERALDS", bid_price, qty))
            buy_cap -= qty

        if ask_size > 0 and sell_cap > 0:
            qty = min(ask_size, sell_cap)
            orders.append(Order("EMERALDS", ask_price, -qty))
            sell_cap -= qty

        # Backup deeper levels using leftover capacity (still fixed offset
        # from fair — these are catch-the-sweep orders, not penny quotes).
        if buy_cap > 0:
            orders.append(Order("EMERALDS", fair - EMERALDS_BACKUP_OFFSET - inv_skew, buy_cap))
        if sell_cap > 0:
            orders.append(Order("EMERALDS", fair + EMERALDS_BACKUP_OFFSET - inv_skew, -sell_cap))

        return orders

    # ═══════════════════════════════════════════════════════
    #  TOMATOES — Drift + Vol Regime Aware MM
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
        thresh = TOMATOES_IMB_THRESH

        instant_fair = 0.5 * wall_mid + 0.5 * (mp if mp is not None else wall_mid)

        if self.tomatoes_ema is None:
            self.tomatoes_ema = instant_fair
        else:
            self.tomatoes_ema = (TOMATOES_EMA_ALPHA * instant_fair
                                 + (1 - TOMATOES_EMA_ALPHA) * self.tomatoes_ema)

        # Track rolling history for vol regime
        self.tomatoes_mid_history.append(wall_mid)
        if len(self.tomatoes_mid_history) > TOMATOES_VOL_WINDOW:
            self.tomatoes_mid_history = self.tomatoes_mid_history[-TOMATOES_VOL_WINDOW:]

        # Rolling std of wall_mid
        vol = 0.0
        if len(self.tomatoes_mid_history) >= 5:
            mean = sum(self.tomatoes_mid_history) / len(self.tomatoes_mid_history)
            var = sum((x - mean) ** 2 for x in self.tomatoes_mid_history) / len(self.tomatoes_mid_history)
            vol = math.sqrt(var)

        is_fast_regime = vol > TOMATOES_VOL_HIGH

        # Drift signal: wall is moving away from EMA → trend
        drift = wall_mid - self.tomatoes_ema
        is_drifting = abs(drift) > TOMATOES_DRIFT_THRESH

        fair = round(self.tomatoes_ema)

        # In fast regime, shrink position cap to limit blast radius
        active_max_pos = TOMATOES_MAX_POS // 2 if is_fast_regime else TOMATOES_MAX_POS

        # ── PHASE 1: TAKE ────────────────────────────────
        for ask_price in sorted(depth.sell_orders.keys()):
            edge = fair - ask_price
            vol_at_lvl = abs(depth.sell_orders[ask_price])
            # Loosen for: confirmed imbalance, OR small inside-wall probes
            min_edge = 1 if (imb > thresh or vol_at_lvl <= SMALL_ORDER_VOL) else 2
            # When drifting down, require more edge before buying
            if is_drifting and drift < 0:
                min_edge += 1
            if edge >= min_edge and buy_cap > 0:
                qty = min(vol_at_lvl, buy_cap, max(0, active_max_pos - pos))
                if qty > 0:
                    orders.append(Order("TOMATOES", ask_price, qty))
                    buy_cap -= qty
                    pos += qty

        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            edge = bid_price - fair
            vol_at_lvl = depth.buy_orders[bid_price]
            min_edge = 1 if (imb < -thresh or vol_at_lvl <= SMALL_ORDER_VOL) else 2
            if is_drifting and drift > 0:
                min_edge += 1
            if edge >= min_edge and sell_cap > 0:
                qty = min(vol_at_lvl, sell_cap, max(0, active_max_pos + pos))
                if qty > 0:
                    orders.append(Order("TOMATOES", bid_price, -qty))
                    sell_cap -= qty
                    pos -= qty

        # ── PHASE 2: FLATTEN ─────────────────────────────
        # More aggressive flatten in fast regime
        flatten_threshold = 3 if is_fast_regime else 5
        if abs(pos) > flatten_threshold:
            if pos > 0 and sell_cap > 0:
                flatten_px = fair if is_fast_regime else fair + 1
                flatten_qty = min(pos - 2, sell_cap)
                if flatten_qty > 0:
                    orders.append(Order("TOMATOES", flatten_px, -flatten_qty))
                    sell_cap -= flatten_qty
                    pos -= flatten_qty
            elif pos < 0 and buy_cap > 0:
                flatten_px = fair if is_fast_regime else fair - 1
                flatten_qty = min(-pos - 2, buy_cap)
                if flatten_qty > 0:
                    orders.append(Order("TOMATOES", flatten_px, flatten_qty))
                    buy_cap -= flatten_qty
                    pos += flatten_qty

        # ── PHASE 3: MAKE ────────────────────────────────
        inv_ratio = pos / max(active_max_pos, 1)
        extra_spread = round(abs(inv_ratio) * TOMATOES_INV_PENALTY * 2)

        # Vol-aware base spread: widen during fast regime
        vol_extra = 2 if is_fast_regime else 0

        bid_offset = TOMATOES_BASE_SPREAD + extra_spread + vol_extra
        ask_offset = TOMATOES_BASE_SPREAD + extra_spread + vol_extra

        inv_skew = round(inv_ratio * 2)
        imb_skew = round(imb * TOMATOES_IMB_SKEW)

        # Drift bias: if wall is trending away from EMA, lean hard with the trend
        # (positive drift = wall above EMA = price rising → raise both quotes)
        drift_skew = 0
        if is_drifting:
            drift_skew = -2 if drift > 0 else 2  # negative = lift quotes up

        bid_price = wall_bid + bid_offset - inv_skew + imb_skew - drift_skew
        ask_price = wall_ask - ask_offset - inv_skew + imb_skew - drift_skew

        bid_price = min(bid_price, fair - 1)
        ask_price = max(ask_price, fair + 1)
        if bid_price >= ask_price:
            bid_price = fair - 2
            ask_price = fair + 2

        max_new_buy = max(0, active_max_pos - max(pos, 0))
        max_new_sell = max(0, active_max_pos + min(pos, 0))

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
        if not depth.buy_orders or not depth.sell_orders:
            return None, None

        max_bv = max(depth.buy_orders.values())
        wall_bid = max(p for p, v in depth.buy_orders.items() if v == max_bv)

        max_av = min(depth.sell_orders.values())
        wall_ask = min(p for p, v in depth.sell_orders.items() if v == max_av)

        return wall_bid, wall_ask
