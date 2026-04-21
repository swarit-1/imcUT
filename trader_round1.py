from datamodel import OrderDepth, UserId, TradingState, Order
import json
import math


class Trader:
    """
    IMC Prosperity 4 — Round 1 (v14)

    ACO: adaptive fair value with hardcoded warmup.
    For the first ACO_WARMUP ticks we use the hardcoded anchor (10000).
    After warmup, if the rolling mean of mids drifts more than
    ACO_ADAPT_THRESHOLD away from the anchor, we switch to the rolling
    mean. Otherwise we stick with the anchor. This gives us insurance
    if the eval day's fair shifts (10000 is a strong prior but not law).

    IPR: slope-aware buy-and-hold with regime detection.
    """

    LIMITS = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    # ACO
    ACO_FAIR = 10000           # hardcoded anchor (used during warmup)
    ACO_INV_SKEW = 0.02        # per-unit fair shift to manage inventory
    ACO_TAKE_EDGE = 2          # take any order at least 2 ticks beyond fair
    ACO_POST_EDGE = 1          # passive quotes 1 tick inside fair
    ACO_WARMUP = 500           # ticks of hardcoded fair before adapting
    ACO_WINDOW = 2000          # rolling window for adaptive mean
    ACO_ADAPT_THRESHOLD = 5    # only override anchor if |mean - 10000| > 5

    def _load(self, td):
        if td and td.strip():
            try:
                return json.loads(td)
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def _save(self, s):
        return json.dumps(s)

    def run(self, state: TradingState):
        result = {}
        ps = self._load(state.traderData)

        # ═══ ACO: mean-revert against adaptive fair (anchored at 10000) ═══
        sym = "ASH_COATED_OSMIUM"
        if sym in state.order_depths:
            od = state.order_depths[sym]
            pos = state.position.get(sym, 0)
            limit = self.LIMITS[sym]

            best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
            best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

            # Day reset: if timestamp went backwards, clear buffer.
            aco_last_ts = ps.get("aco_last_ts", -1)
            if state.timestamp < aco_last_ts:
                ps["aco_buf_mid"] = []
            ps["aco_last_ts"] = state.timestamp

            # Track rolling mid for adaptive fair.
            aco_mid = None
            if best_bid is not None and best_ask is not None:
                aco_mid = (best_bid + best_ask) / 2
            elif best_bid is not None:
                aco_mid = float(best_bid)
            elif best_ask is not None:
                aco_mid = float(best_ask)

            aco_buf = ps.get("aco_buf_mid", [])
            if aco_mid is not None:
                aco_buf.append(aco_mid)
                if len(aco_buf) > self.ACO_WINDOW:
                    aco_buf = aco_buf[-self.ACO_WINDOW:]
            ps["aco_buf_mid"] = aco_buf

            # Adaptive anchor: use hardcoded 10000 during warmup, then
            # switch to rolling mean only if it has drifted significantly.
            anchor = self.ACO_FAIR
            if len(aco_buf) >= self.ACO_WARMUP:
                rolling_mean = sum(aco_buf) / len(aco_buf)
                if abs(rolling_mean - self.ACO_FAIR) > self.ACO_ADAPT_THRESHOLD:
                    anchor = rolling_mean

            # Inventory-skewed fair: nudges fair against current inventory so
            # we lean toward flattening instead of doubling down at extremes.
            fair = anchor - self.ACO_INV_SKEW * pos

            orders = []
            cp = pos  # running projected position after our take orders fill

            # ─── Take all asks priced strictly below fair ───
            # Walk levels from cheapest up; stop once the price is no longer a buy.
            if od.sell_orders:
                for ask_p in sorted(od.sell_orders.keys()):
                    if ask_p <= fair - self.ACO_TAKE_EDGE:
                        avail = abs(od.sell_orders[ask_p])
                        room = limit - cp
                        qty = min(avail, room)
                        if qty > 0:
                            orders.append(Order(sym, ask_p, qty))
                            cp += qty
                        if cp >= limit:
                            break
                    else:
                        break

            # ─── Take all bids priced strictly above fair ───
            if od.buy_orders:
                for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_p >= fair + self.ACO_TAKE_EDGE:
                        avail = od.buy_orders[bid_p]
                        room = limit + cp
                        qty = min(avail, room)
                        if qty > 0:
                            orders.append(Order(sym, bid_p, -qty))
                            cp -= qty
                        if cp <= -limit:
                            break
                    else:
                        break

            # Recompute skewed fair using projected position post-take.
            af = anchor - self.ACO_INV_SKEW * cp

            # ─── Passive pennying around fair ───
            # Bid one tick inside the bot spread, but never above fair-1
            # (we never want to passively buy at or above fair).
            if best_bid is not None:
                our_bid = min(best_bid + 1, math.floor(af) - self.ACO_POST_EDGE + 1)
                # max additional buys we can post without breaking limit
                buy_room = max(0, limit - cp)
                # Clamp size to limit (fixes "exceeded limit of 80" warning
                # that fired when cp was deeply negative and limit-cp > 80).
                sz = min(buy_room, limit)
                if sz > 0 and our_bid < fair:
                    orders.append(Order(sym, our_bid, sz))

            if best_ask is not None:
                our_ask = max(best_ask - 1, math.ceil(af) + self.ACO_POST_EDGE - 1)
                sell_room = max(0, limit + cp)
                sz = min(sell_room, limit)
                if sz > 0 and our_ask > fair:
                    orders.append(Order(sym, our_ask, -sz))

            result[sym] = orders

                # ═══ IPR: slope-aware directional with regime detection ═══
        sym = "INTARIAN_PEPPER_ROOT"
        if sym in state.order_depths:
            od = state.order_depths[sym]
            pos = state.position.get(sym, 0)
            limit = self.LIMITS[sym]

            best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
            best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

            # Only compute mid from real quotes — skip empty book ticks
            has_book = best_bid is not None or best_ask is not None
            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid = float(best_bid)
            elif best_ask is not None:
                mid = float(best_ask)
            else:
                mid = None  # truly empty book — skip slope update

            # Slope estimation
            WINDOW = 200
            CONFIRM_UP = 0.0003
            CONFIRM_DOWN = -0.0003

            last_ts = ps.get("ipr_last_ts", -1)
            if state.timestamp < last_ts:
                ps["ipr_buf_ts"] = []
                ps["ipr_buf_mid"] = []
                ps["ipr_regime"] = None
            ps["ipr_last_ts"] = state.timestamp

            buf_ts = ps.get("ipr_buf_ts", [])
            buf_mid = ps.get("ipr_buf_mid", [])

            # Only add to buffer if we have a real mid price
            if mid is not None:
                buf_ts.append(state.timestamp)
                buf_mid.append(mid)
                if len(buf_ts) > WINDOW:
                    buf_ts = buf_ts[-WINDOW:]
                    buf_mid = buf_mid[-WINDOW:]
            ps["ipr_buf_ts"] = buf_ts
            ps["ipr_buf_mid"] = buf_mid

            slope = None
            if len(buf_ts) >= 50:
                n = len(buf_ts)
                sx = sum(buf_ts); sy = sum(buf_mid)
                sxy = sum(x * y for x, y in zip(buf_ts, buf_mid))
                sx2 = sum(x * x for x in buf_ts)
                d = n * sx2 - sx * sx
                if abs(d) > 1e-10:
                    slope = (n * sxy - sx * sy) / d

            # Regime with hysteresis
            regime = ps.get("ipr_regime")
            if slope is not None:
                if slope > CONFIRM_UP:
                    regime = "UP"
                elif slope < CONFIRM_DOWN:
                    regime = "DOWN"
            if regime is None:
                regime = "UP"
            ps["ipr_regime"] = regime

            if regime == "UP":
                target = limit
            elif regime == "DOWN":
                target = -limit
            else:
                target = 0

            needed = target - pos
            orders = []

            if needed > 0:
                remaining = needed
                if od.sell_orders:
                    for ask_p in sorted(od.sell_orders.keys()):
                        vol = abs(od.sell_orders[ask_p])
                        qty = min(vol, remaining)
                        if qty > 0:
                            orders.append(Order(sym, ask_p, qty))
                            remaining -= qty
                        if remaining <= 0:
                            break
                if remaining > 0 and best_ask is not None:
                    orders.append(Order(sym, best_ask, remaining))
                elif remaining > 0 and best_bid is not None:
                    orders.append(Order(sym, best_bid + 1, remaining))

            elif needed < 0:
                remaining = abs(needed)
                if od.buy_orders:
                    for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                        vol = od.buy_orders[bid_p]
                        qty = min(vol, remaining)
                        if qty > 0:
                            orders.append(Order(sym, bid_p, -qty))
                            remaining -= qty
                        if remaining <= 0:
                            break
                if remaining > 0 and best_bid is not None:
                    orders.append(Order(sym, best_bid, -remaining))
                elif remaining > 0 and best_ask is not None:
                    orders.append(Order(sym, best_ask - 1, -remaining))

            result[sym] = orders

        return result, 0, self._save(ps)