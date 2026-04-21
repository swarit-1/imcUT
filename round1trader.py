from datamodel import OrderDepth, UserId, TradingState, Order
import json
import math


class Trader:
    LIMITS = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    # ACO Base Parameters
    ACO_FAIR_ANCHOR = 10000    # Hardcoded fallback fair
    ACO_INV_SKEW = 0.02        # Per-unit fair shift to manage inventory
    ACO_WARMUP = 50            # Ticks needed before OU regression starts
    ACO_WINDOW = 300           # Rolling window for AR(1) regression

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

        # ═══ ACO: OU Stochastic Process Calibration ═══
        sym = "ASH_COATED_OSMIUM"
        if sym in state.order_depths:
            od = state.order_depths[sym]
            pos = state.position.get(sym, 0)
            limit = self.LIMITS[sym]

            best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
            best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

            # Day reset buffer clearing
            aco_last_ts = ps.get("aco_last_ts", -1)
            if state.timestamp < aco_last_ts:
                ps["aco_buf_mid"] = []
            ps["aco_last_ts"] = state.timestamp

            # Calculate current mid
            mid = None
            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid = float(best_bid)
            elif best_ask is not None:
                mid = float(best_ask)

            buf_mid = ps.get("aco_buf_mid", [])
            if mid is not None:
                buf_mid.append(mid)
                if len(buf_mid) > self.ACO_WINDOW:
                    buf_mid = buf_mid[-self.ACO_WINDOW:]
            ps["aco_buf_mid"] = buf_mid

            # --- Real-Time OU Calibration via AR(1) ---
            theta = self.ACO_FAIR_ANCHOR
            dynamic_take_edge = 2.0
            dynamic_post_edge = 1.0

            n = len(buf_mid) - 1
            if n >= self.ACO_WARMUP:
                # X is S_{t-1}, Y is S_t
                sum_x = sum(buf_mid[:-1])
                sum_y = sum(buf_mid[1:])
                sum_x2 = sum(x * x for x in buf_mid[:-1])
                sum_xy = sum(x * y for x, y in zip(buf_mid[:-1], buf_mid[1:]))

                denominator = (n * sum_x2 - sum_x ** 2)

                if denominator > 1e-8:
                    beta = (n * sum_xy - sum_x * sum_y) / denominator
                    alpha = (sum_y - beta * sum_x) / n

                    # Ensure process is mean-reverting (0 < beta < 1).
                    # Upper-bound tightened to 0.9999 so alpha/(1-beta) cannot blow up.
                    if 0 < beta < 0.9999:
                        theta = alpha / (1 - beta)

                        # Residual variance (sigma proxy)
                        residuals = [buf_mid[i + 1] - (alpha + beta * buf_mid[i]) for i in range(n)]
                        var_res = sum(r * r for r in residuals) / n
                        sigma = math.sqrt(var_res)

                        # Dynamic spreads based on volatility
                        dynamic_take_edge = max(1.0, math.ceil(sigma * 0.8))
                        dynamic_post_edge = max(1.0, math.floor(sigma * 0.4))
                    else:
                        # Non-stationary: fall back to anchor, NOT mid
                        # (mid could be None on empty book, and even if present,
                        # anchoring to current mid gives zero trading edge).
                        theta = self.ACO_FAIR_ANCHOR

            # --- Inventory Skewing ---
            fair = theta - self.ACO_INV_SKEW * pos

            orders = []
            cp = pos

            # --- Take Asks (Aggressive Buy) ---
            if od.sell_orders:
                for ask_p in sorted(od.sell_orders.keys()):
                    if ask_p <= fair - dynamic_take_edge:
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

            # --- Take Bids (Aggressive Sell) ---
            if od.buy_orders:
                for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_p >= fair + dynamic_take_edge:
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

            # Recompute skewed fair using projected position post-take
            af = theta - self.ACO_INV_SKEW * cp

            # --- Passive Pennying (Maker) ---
            # Cast order prices to int — Order expects integer prices, and
            # math.floor(af) - float_edge + 1 would otherwise yield a float.
            if best_bid is not None:
                our_bid = int(min(best_bid + 1, math.floor(af) - dynamic_post_edge + 1))
                sz = min(max(0, limit - cp), limit)
                if sz > 0 and our_bid < fair:
                    orders.append(Order(sym, our_bid, sz))

            if best_ask is not None:
                our_ask = int(max(best_ask - 1, math.ceil(af) + dynamic_post_edge - 1))
                sz = min(max(0, limit + cp), limit)
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
            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid = float(best_bid)
            elif best_ask is not None:
                mid = float(best_ask)
            else:
                mid = None

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
            buf_mid_ipr = ps.get("ipr_buf_mid", [])

            if mid is not None:
                buf_ts.append(state.timestamp)
                buf_mid_ipr.append(mid)
                if len(buf_ts) > WINDOW:
                    buf_ts = buf_ts[-WINDOW:]
                    buf_mid_ipr = buf_mid_ipr[-WINDOW:]
            ps["ipr_buf_ts"] = buf_ts
            ps["ipr_buf_mid"] = buf_mid_ipr

            slope = None
            if len(buf_ts) >= 50:
                n = len(buf_ts)
                sx = sum(buf_ts)
                sy = sum(buf_mid_ipr)
                sxy = sum(x * y for x, y in zip(buf_ts, buf_mid_ipr))
                sx2 = sum(x * x for x in buf_ts)
                d = n * sx2 - sx * sx
                if abs(d) > 1e-10:
                    slope = (n * sxy - sx * sy) / d

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
