from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import math


class Trader:
    """
    Round 0 strategy for Prosperity 4.

    Core ideas:
    - EMERALDS behaves like a stationary product around 10_000, so we market make
      aggressively inside the spread with inventory-aware skew.
    - TOMATOES is not fixed; it drifts slowly. We estimate a short-horizon fair value
      using the current mid, the last few mids, and a microprice signal, then market
      make around that fair value.
    - For both products we first take obviously favorable quotes, then post passive
      orders one tick inside the current spread whenever that is still better than our
      minimum edge requirement.
    """

    LIMITS: Dict[str, int] = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    EMERALDS_FAIR = 10_000.0
    EMERALDS_TAKE_EDGE = 1.0
    EMERALDS_QUOTE_EDGE = 1.0
    EMERALDS_INV_SKEW = 0.05

    TOMATOES_BASE = 5_000.0
    # Fitted on the public round-0 data: current mid + 3 lagged mids.
    TOMATOES_AR_COEFS = [0.5304, 0.2184, 0.1416, 0.1063]
    TOMATOES_SIGNAL_COEF = 0.7753
    TOMATOES_TAKE_EDGE = 1.0
    TOMATOES_QUOTE_EDGE = 2.0
    TOMATOES_INV_SKEW = 0.08

    MAX_HISTORY = 4

    def run(self, state: TradingState):
        cache = self._load_cache(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, depth in state.order_depths.items():
            position = state.position.get(product, 0)
            orders: List[Order] = []

            if product == "EMERALDS":
                orders = self._trade_emeralds(depth, position)
            elif product == "TOMATOES":
                history = cache.get("TOMATOES", [])
                orders = self._trade_tomatoes(depth, position, history)
            else:
                orders = []

            result[product] = orders

        # Update cache with the mids from the current state so the next invocation has history.
        for product, depth in state.order_depths.items():
            best_bid, best_ask, _, _, mid, _ = self._top_of_book(depth)
            if best_bid is None or best_ask is None or mid is None:
                continue
            history = cache.setdefault(product, [])
            history.append(mid)
            if len(history) > self.MAX_HISTORY:
                cache[product] = history[-self.MAX_HISTORY :]

        trader_data = self._dump_cache(cache)
        conversions = 0
        return result, conversions, trader_data

    def _trade_emeralds(self, depth: OrderDepth, position: int) -> List[Order]:
        best_bid, best_ask, bid_orders, ask_orders, _, _ = self._top_of_book(depth)
        if best_bid is None or best_ask is None:
            return []

        reservation = self.EMERALDS_FAIR - self.EMERALDS_INV_SKEW * position
        limit = self.LIMITS["EMERALDS"]
        buy_cap = max(0, limit - position)
        sell_cap = max(0, limit + position)
        orders: List[Order] = []

        # Step 1: take stale / favorable quotes.
        for ask_price, ask_volume in ask_orders:
            if buy_cap <= 0:
                break
            if ask_price <= reservation - self.EMERALDS_TAKE_EDGE:
                qty = min(buy_cap, ask_volume)
                if qty > 0:
                    orders.append(Order("EMERALDS", ask_price, qty))
                    buy_cap -= qty
            else:
                break

        for bid_price, bid_volume in bid_orders:
            if sell_cap <= 0:
                break
            if bid_price >= reservation + self.EMERALDS_TAKE_EDGE:
                qty = min(sell_cap, bid_volume)
                if qty > 0:
                    orders.append(Order("EMERALDS", bid_price, -qty))
                    sell_cap -= qty
            else:
                break

        # Step 2: passive quotes. Quote one tick inside the spread whenever that still
        # leaves at least our minimum edge versus reservation price.
        bid_px, ask_px = self._passive_quote_prices(
            reservation,
            best_bid,
            best_ask,
            self.EMERALDS_QUOTE_EDGE,
        )

        if buy_cap > 0:
            orders.append(Order("EMERALDS", bid_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order("EMERALDS", ask_px, -sell_cap))

        return orders

    def _trade_tomatoes(self, depth: OrderDepth, position: int, history: List[float]) -> List[Order]:
        best_bid, best_ask, bid_orders, ask_orders, mid, micro = self._top_of_book(depth)
        if best_bid is None or best_ask is None or mid is None or micro is None:
            return []

        fair = self._tomatoes_fair_value(mid, micro, history)
        reservation = fair - self.TOMATOES_INV_SKEW * position
        limit = self.LIMITS["TOMATOES"]
        buy_cap = max(0, limit - position)
        sell_cap = max(0, limit + position)
        orders: List[Order] = []

        # Step 1: hit obviously favorable prices across visible levels.
        for ask_price, ask_volume in ask_orders:
            if buy_cap <= 0:
                break
            if ask_price <= reservation - self.TOMATOES_TAKE_EDGE:
                qty = min(buy_cap, ask_volume)
                if qty > 0:
                    orders.append(Order("TOMATOES", ask_price, qty))
                    buy_cap -= qty
            else:
                break

        for bid_price, bid_volume in bid_orders:
            if sell_cap <= 0:
                break
            if bid_price >= reservation + self.TOMATOES_TAKE_EDGE:
                qty = min(sell_cap, bid_volume)
                if qty > 0:
                    orders.append(Order("TOMATOES", bid_price, -qty))
                    sell_cap -= qty
            else:
                break

        # Step 2: inside-spread market making around the forecast fair value.
        bid_px, ask_px = self._passive_quote_prices(
            reservation,
            best_bid,
            best_ask,
            self.TOMATOES_QUOTE_EDGE,
        )

        if buy_cap > 0:
            orders.append(Order("TOMATOES", bid_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order("TOMATOES", ask_px, -sell_cap))

        return orders

    def _tomatoes_fair_value(self, current_mid: float, current_micro: float, history: List[float]) -> float:
        base = self.TOMATOES_BASE
        signal = current_micro - current_mid

        m1 = history[-1] if len(history) >= 1 else current_mid
        m2 = history[-2] if len(history) >= 2 else m1
        m3 = history[-3] if len(history) >= 3 else m2

        x0 = current_mid - base
        x1 = m1 - base
        x2 = m2 - base
        x3 = m3 - base

        fair = (
            base
            + self.TOMATOES_AR_COEFS[0] * x0
            + self.TOMATOES_AR_COEFS[1] * x1
            + self.TOMATOES_AR_COEFS[2] * x2
            + self.TOMATOES_AR_COEFS[3] * x3
            + self.TOMATOES_SIGNAL_COEF * signal
        )
        return fair

    def _passive_quote_prices(
        self,
        reservation: float,
        best_bid: int,
        best_ask: int,
        min_edge: float,
    ) -> Tuple[int, int]:
        # Safe prices from our fair value.
        safe_bid = int(math.floor(reservation - min_edge))
        safe_ask = int(math.ceil(reservation + min_edge))

        # Best passive prices that are still one tick inside the spread when possible.
        if best_bid + 1 <= safe_bid:
            bid_px = best_bid + 1
        else:
            bid_px = safe_bid

        if best_ask - 1 >= safe_ask:
            ask_px = best_ask - 1
        else:
            ask_px = safe_ask

        # Final guards against crossing due to extreme skew or narrow spreads.
        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)

        return bid_px, ask_px

    def _top_of_book(
        self,
        depth: OrderDepth,
    ) -> Tuple[int | None, int | None, List[Tuple[int, int]], List[Tuple[int, int]], float | None, float | None]:
        if not depth.buy_orders or not depth.sell_orders:
            return None, None, [], [], None, None

        bid_orders = sorted(
            [(int(price), int(volume)) for price, volume in depth.buy_orders.items()],
            key=lambda x: -x[0],
        )
        ask_orders = sorted(
            [(int(price), int(abs(volume))) for price, volume in depth.sell_orders.items()],
            key=lambda x: x[0],
        )

        best_bid, best_bid_volume = bid_orders[0]
        best_ask, best_ask_volume = ask_orders[0]
        mid = 0.5 * (best_bid + best_ask)

        total_top_volume = best_bid_volume + best_ask_volume
        if total_top_volume > 0:
            micro = (best_bid * best_ask_volume + best_ask * best_bid_volume) / total_top_volume
        else:
            micro = mid

        return best_bid, best_ask, bid_orders, ask_orders, mid, micro

    def _load_cache(self, trader_data: str) -> Dict[str, List[float]]:
        cache: Dict[str, List[float]] = {
            "EMERALDS": [],
            "TOMATOES": [],
        }
        if not trader_data:
            return cache

        try:
            for block in trader_data.split("|"):
                if not block or ":" not in block:
                    continue
                product, raw_values = block.split(":", 1)
                if product not in cache:
                    continue
                if raw_values:
                    values = []
                    for item in raw_values.split(","):
                        if item:
                            values.append(float(item))
                    cache[product] = values[-self.MAX_HISTORY :]
            return cache
        except Exception:
            return {
                "EMERALDS": [],
                "TOMATOES": [],
            }

    def _dump_cache(self, cache: Dict[str, List[float]]) -> str:
        parts: List[str] = []
        for product in ["EMERALDS", "TOMATOES"]:
            history = cache.get(product, [])[-self.MAX_HISTORY :]
            serialized = ",".join(f"{value:.2f}" for value in history)
            parts.append(f"{product}:{serialized}")
        return "|".join(parts)
