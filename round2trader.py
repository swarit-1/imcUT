from datamodel import OrderDepth, UserId, TradingState, Order
import json
import math


class Trader:
    """
    IMC Prosperity 4 — Round 2 (v16, safety-hardened on v15)

    Core strategies preserved:
      ACO: mean-revert around 10000 with adaptive anchor, skew 0.05
      IPR: slope-aware directional follower with patient entry

    Defensive layers added:
      Per-product realized+unrealized PnL (cash + pos * mid)
      Soft stop  -2500  -> staged passive->cross liquidation, then halt
      Hard stop  -4000  -> cross-spread flatten, halt product
      Global kill -5000 -> liquidate both products, halt all; kill flag persists
      IPR cap lowered 80 -> 50
      IPR fast/slow slope via EMA (a=0.067 / a=0.01); 3-tick fast-disagreement halves target
      Entry throttle: <=20 units/tick in the adding-to-position direction
      ACO per-side sizing bugfix (uses original_pos, not cp, for quote capacity)
      Defensive empty-book handling
    """

    LIMITS = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 50}
    EXCHANGE_LIMITS = {"ASH_COATED_OSMIUM": 80, "INTARIAN_PEPPER_ROOT": 80}

    SOFT_STOP_PNL = -2500
    HARD_STOP_PNL = -4000
    GLOBAL_KILL_PNL = -5000

    LIQ_PASSIVE_TICKS = 10
    LIQ_CROSS1_TICKS = 20
    LIQ_REDUCE_T1 = 0.50
    LIQ_REDUCE_T2 = 0.80

    MAX_ENTRY_PER_TICK = 20

    ACO_FAIR = 10000
    ACO_INV_SKEW = 0.05
    ACO_TAKE_EDGE = 2
    ACO_POST_EDGE = 1
    ACO_ADAPT_WARMUP = 500
    ACO_ADAPT_WINDOW = 2000
    ACO_ADAPT_THRESH = 5

    IPR_SLOW_ALPHA = 0.01
    IPR_FAST_ALPHA = 0.067
    IPR_CONFIRM_UP = 0.0006
    IPR_CONFIRM_DOWN = -0.0006
    IPR_WARMUP = 50
    IPR_ENTRY_TAKE_EDGE = 2
    IPR_SELL_QUOTE_OFFSET = 5
    IPR_SELL_QUOTE_SIZE = 20
    IPR_FAST_DISAGREE_THRESH = 3

    def _load(self, td):
        if td and td.strip():
            try:
                return json.loads(td)
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def _save(self, s):
        return json.dumps(s)

    def _book_sides(self, od):
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
        else:
            mid = None
        return best_bid, best_ask, mid

    def _update_cash(self, ps, cash_key, trades):
        for t in trades or []:
            b, s = getattr(t, "buyer", None), getattr(t, "seller", None)
            we_bought = b in ("SUBMISSION", "")
            we_sold = s in ("SUBMISSION", "")
            if we_bought and not we_sold:
                ps[cash_key] -= t.price * t.quantity
            elif we_sold and not we_bought:
                ps[cash_key] += t.price * t.quantity

    def _pnl(self, cash, pos, mid):
        if mid is None:
            # Conservative: unknown mid, mark at cash only; pos risk uncounted this tick.
            return cash
        return cash + pos * mid

    def _throttle(self, orders, pos, max_entry):
        """Cap orders that grow |pos| to <= max_entry/tick cumulatively; exits uncapped."""
        buy_cap = max_entry if pos >= 0 else 10**9
        sell_cap = max_entry if pos <= 0 else 10**9
        bu, su = 0, 0
        out = []
        for o in orders:
            if o.quantity > 0:
                rem = buy_cap - bu
                if rem <= 0:
                    continue
                q = min(o.quantity, rem)
                if q > 0:
                    out.append(Order(o.symbol, o.price, q))
                    bu += q
            elif o.quantity < 0:
                rem = sell_cap - su
                if rem <= 0:
                    continue
                q = min(-o.quantity, rem)
                if q > 0:
                    out.append(Order(o.symbol, o.price, -q))
                    su += q
        return out

    def _flatten(self, sym, pos, best_bid, best_ask, ticks_through):
        """Cross spread by `ticks_through` tick(s) to flatten pos in one shot."""
        if pos == 0 or best_bid is None or best_ask is None:
            return []
        if pos > 0:
            price = best_bid - (ticks_through - 1)
            return [Order(sym, price, -pos)]
        else:
            price = best_ask + (ticks_through - 1)
            return [Order(sym, price, -pos)]

    def _liq_orders(self, sym, pos, best_bid, best_ask, liq_ticks, initial_abs):
        if pos == 0 or best_bid is None or best_ask is None:
            return []
        cur_abs = abs(pos)
        reduction = (initial_abs - cur_abs) / max(1, initial_abs)

        if liq_ticks < self.LIQ_PASSIVE_TICKS:
            if pos > 0:
                price = best_ask - 1 if best_ask - 1 > best_bid else best_bid + 1
                return [Order(sym, price, -pos)]
            else:
                price = best_bid + 1 if best_bid + 1 < best_ask else best_ask - 1
                return [Order(sym, price, -pos)]

        if liq_ticks < self.LIQ_CROSS1_TICKS:
            if reduction < self.LIQ_REDUCE_T1:
                half = max(1, cur_abs // 2)
                if pos > 0:
                    return [Order(sym, best_bid, -half)]
                else:
                    return [Order(sym, best_ask, half)]
            if pos > 0:
                price = best_ask - 1 if best_ask - 1 > best_bid else best_bid + 1
                return [Order(sym, price, -pos)]
            else:
                price = best_bid + 1 if best_bid + 1 < best_ask else best_ask - 1
                return [Order(sym, price, -pos)]

        if reduction < self.LIQ_REDUCE_T2:
            if pos > 0:
                return [Order(sym, best_bid - 1, -cur_abs)]
            else:
                return [Order(sym, best_ask + 1, cur_abs)]
        if pos > 0:
            price = best_ask - 1 if best_ask - 1 > best_bid else best_bid + 1
            return [Order(sym, price, -pos)]
        else:
            price = best_bid + 1 if best_bid + 1 < best_ask else best_ask - 1
            return [Order(sym, price, -pos)]

    def run(self, state: TradingState):
        result = {}
        ps = self._load(state.traderData)

        # Kill switch persists even if buffers reset on timestamp wrap.
        global_kill = bool(ps.get("global_kill", False))

        last_seen_ts = ps.get("last_seen_ts", -1)
        if state.timestamp < last_seen_ts:
            ps = {"global_kill": global_kill}
        ps["last_seen_ts"] = state.timestamp

        ps.setdefault("aco_cash", 0.0)
        ps.setdefault("ipr_cash", 0.0)
        ps.setdefault("aco_halted", False)
        ps.setdefault("ipr_halted", False)
        ps.setdefault("aco_liq_mode", False)
        ps.setdefault("ipr_liq_mode", False)
        ps.setdefault("aco_liq_ticks", 0)
        ps.setdefault("ipr_liq_ticks", 0)
        ps.setdefault("aco_liq_initial_abs", 0)
        ps.setdefault("ipr_liq_initial_abs", 0)
        ps.setdefault("global_kill", global_kill)

        self._update_cash(ps, "aco_cash", state.own_trades.get("ASH_COATED_OSMIUM", []))
        self._update_cash(ps, "ipr_cash", state.own_trades.get("INTARIAN_PEPPER_ROOT", []))

        aco_pos = state.position.get("ASH_COATED_OSMIUM", 0)
        ipr_pos = state.position.get("INTARIAN_PEPPER_ROOT", 0)
        aco_od = state.order_depths.get("ASH_COATED_OSMIUM")
        ipr_od = state.order_depths.get("INTARIAN_PEPPER_ROOT")

        aco_bb = aco_ba = aco_mid = None
        ipr_bb = ipr_ba = ipr_mid = None
        if aco_od is not None:
            aco_bb, aco_ba, aco_mid = self._book_sides(aco_od)
        if ipr_od is not None:
            ipr_bb, ipr_ba, ipr_mid = self._book_sides(ipr_od)

        aco_pnl = self._pnl(ps["aco_cash"], aco_pos, aco_mid)
        ipr_pnl = self._pnl(ps["ipr_cash"], ipr_pos, ipr_mid)
        total_pnl = aco_pnl + ipr_pnl

        if not ps["global_kill"] and total_pnl < self.GLOBAL_KILL_PNL:
            ps["global_kill"] = True

        if ps["global_kill"]:
            if aco_pos != 0 and aco_bb is not None and aco_ba is not None:
                result["ASH_COATED_OSMIUM"] = self._flatten(
                    "ASH_COATED_OSMIUM", aco_pos, aco_bb, aco_ba, 2)
            if ipr_pos != 0 and ipr_bb is not None and ipr_ba is not None:
                result["INTARIAN_PEPPER_ROOT"] = self._flatten(
                    "INTARIAN_PEPPER_ROOT", ipr_pos, ipr_bb, ipr_ba, 2)
            return result, 0, self._save(ps)

        if not ps["aco_halted"] and aco_pnl < self.HARD_STOP_PNL:
            ps["aco_halted"] = True
            ps["aco_liq_mode"] = False
        if not ps["ipr_halted"] and ipr_pnl < self.HARD_STOP_PNL:
            ps["ipr_halted"] = True
            ps["ipr_liq_mode"] = False

        if (not ps["aco_halted"] and not ps["aco_liq_mode"]
                and aco_pnl < self.SOFT_STOP_PNL):
            ps["aco_liq_mode"] = True
            ps["aco_liq_ticks"] = 0
            ps["aco_liq_initial_abs"] = abs(aco_pos)
        if (not ps["ipr_halted"] and not ps["ipr_liq_mode"]
                and ipr_pnl < self.SOFT_STOP_PNL):
            ps["ipr_liq_mode"] = True
            ps["ipr_liq_ticks"] = 0
            ps["ipr_liq_initial_abs"] = abs(ipr_pos)

        # ═══════════════════ ACO ═══════════════════
        sym = "ASH_COATED_OSMIUM"
        if aco_od is not None:
            od = aco_od
            pos = aco_pos
            limit = self.LIMITS[sym]
            best_bid, best_ask, mid = aco_bb, aco_ba, aco_mid

            aco_buf = ps.get("aco_mid_buf", [])
            if mid is not None:
                aco_buf.append(mid)
                if len(aco_buf) > self.ACO_ADAPT_WINDOW:
                    aco_buf = aco_buf[-self.ACO_ADAPT_WINDOW:]
            ps["aco_mid_buf"] = aco_buf

            orders = []
            skip_throttle = False

            if ps["aco_halted"]:
                if pos != 0 and best_bid is not None and best_ask is not None:
                    orders = self._flatten(sym, pos, best_bid, best_ask, 2)
                skip_throttle = True
            elif ps["aco_liq_mode"]:
                if pos == 0:
                    ps["aco_liq_mode"] = False
                    ps["aco_halted"] = True  # "no new positions for rest of run"
                elif best_bid is not None and best_ask is not None:
                    orders = self._liq_orders(
                        sym, pos, best_bid, best_ask,
                        ps["aco_liq_ticks"], ps["aco_liq_initial_abs"])
                    ps["aco_liq_ticks"] += 1
                skip_throttle = True
            elif best_bid is not None and best_ask is not None:
                if len(aco_buf) >= self.ACO_ADAPT_WARMUP:
                    rolling_mean = sum(aco_buf) / len(aco_buf)
                    anchor = (rolling_mean
                              if abs(rolling_mean - self.ACO_FAIR) >= self.ACO_ADAPT_THRESH
                              else self.ACO_FAIR)
                else:
                    anchor = self.ACO_FAIR

                fair = anchor - self.ACO_INV_SKEW * pos
                original_pos = pos
                cp = pos
                take_buy_sum = 0
                take_sell_sum = 0

                if od.sell_orders:
                    for ask_p in sorted(od.sell_orders.keys()):
                        if ask_p <= fair - self.ACO_TAKE_EDGE:
                            avail = abs(od.sell_orders[ask_p])
                            room = limit - cp
                            qty = min(avail, room)
                            if qty > 0:
                                orders.append(Order(sym, ask_p, qty))
                                cp += qty
                                take_buy_sum += qty
                            if cp >= limit:
                                break
                        else:
                            break

                if od.buy_orders:
                    for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                        if bid_p >= fair + self.ACO_TAKE_EDGE:
                            avail = od.buy_orders[bid_p]
                            room = limit + cp
                            qty = min(avail, room)
                            if qty > 0:
                                orders.append(Order(sym, bid_p, -qty))
                                cp -= qty
                                take_sell_sum += qty
                            if cp <= -limit:
                                break
                        else:
                            break

                af = anchor - self.ACO_INV_SKEW * cp

                # Per-side sizing vs ORIGINAL pos — exchange check is sum(side)+|pos_start|<=limit.
                # Using `cp` (post-takes) double-counts freed capacity and caused cap breaches.
                bid_cap = max(0, limit - original_pos - take_buy_sum)
                ask_cap = max(0, limit + original_pos - take_sell_sum)

                our_bid = min(best_bid + 1, math.floor(af) - self.ACO_POST_EDGE + 1)
                if bid_cap > 0 and our_bid < fair:
                    orders.append(Order(sym, our_bid, bid_cap))

                our_ask = max(best_ask - 1, math.ceil(af) + self.ACO_POST_EDGE - 1)
                if ask_cap > 0 and our_ask > fair:
                    orders.append(Order(sym, our_ask, -ask_cap))

            if orders and not skip_throttle:
                orders = self._throttle(orders, pos, self.MAX_ENTRY_PER_TICK)
            if orders:
                result[sym] = orders

        # ═══════════════════ IPR ═══════════════════
        sym = "INTARIAN_PEPPER_ROOT"
        if ipr_od is not None:
            od = ipr_od
            pos = ipr_pos
            limit = self.LIMITS[sym]
            best_bid, best_ask, mid = ipr_bb, ipr_ba, ipr_mid

            prev_mid = ps.get("ipr_prev_mid")
            prev_ts = ps.get("ipr_prev_ts")
            slope_slow = ps.get("ipr_slope_slow", 0.0)
            slope_fast = ps.get("ipr_slope_fast", 0.0)
            tick_count = ps.get("ipr_tick_count", 0)

            if mid is not None and prev_mid is not None and prev_ts is not None:
                dt = state.timestamp - prev_ts
                if dt > 0:
                    # Normalize to slope-per-timestamp-unit so thresholds (0.0006)
                    # match the old rolling-OLS-on-timestamps scale.
                    sample = (mid - prev_mid) / dt
                    a_s = self.IPR_SLOW_ALPHA
                    a_f = self.IPR_FAST_ALPHA
                    slope_slow = a_s * sample + (1 - a_s) * slope_slow
                    slope_fast = a_f * sample + (1 - a_f) * slope_fast
                    tick_count += 1

            if mid is not None:
                ps["ipr_prev_mid"] = mid
                ps["ipr_prev_ts"] = state.timestamp
            ps["ipr_slope_slow"] = slope_slow
            ps["ipr_slope_fast"] = slope_fast
            ps["ipr_tick_count"] = tick_count

            regime = ps.get("ipr_regime")
            if tick_count >= self.IPR_WARMUP:
                if slope_slow > self.IPR_CONFIRM_UP:
                    regime = "UP"
                elif slope_slow < self.IPR_CONFIRM_DOWN:
                    regime = "DOWN"
            if regime is None:
                regime = "UP"
            ps["ipr_regime"] = regime

            disagree = ps.get("ipr_fast_disagree", 0)
            if regime == "UP" and slope_fast < 0:
                disagree += 1
            elif regime == "DOWN" and slope_fast > 0:
                disagree += 1
            else:
                disagree = 0
            ps["ipr_fast_disagree"] = disagree

            fair_est = mid if mid is not None else None

            orders = []
            skip_throttle = False

            if ps["ipr_halted"]:
                if pos != 0 and best_bid is not None and best_ask is not None:
                    orders = self._flatten(sym, pos, best_bid, best_ask, 2)
                skip_throttle = True
            elif ps["ipr_liq_mode"]:
                if pos == 0:
                    ps["ipr_liq_mode"] = False
                    ps["ipr_halted"] = True
                elif best_bid is not None and best_ask is not None:
                    orders = self._liq_orders(
                        sym, pos, best_bid, best_ask,
                        ps["ipr_liq_ticks"], ps["ipr_liq_initial_abs"])
                    ps["ipr_liq_ticks"] += 1
                skip_throttle = True
            elif (best_bid is not None and best_ask is not None
                    and fair_est is not None):
                if regime == "UP":
                    target = limit
                elif regime == "DOWN":
                    target = -limit
                else:
                    target = 0

                if disagree >= self.IPR_FAST_DISAGREE_THRESH:
                    # Fast signal fighting regime — halve exposure without flipping.
                    target = target // 2

                if target > pos:
                    remaining = target - pos
                    if od.sell_orders:
                        for ask_p in sorted(od.sell_orders.keys()):
                            if ask_p <= fair_est + self.IPR_ENTRY_TAKE_EDGE:
                                vol = abs(od.sell_orders[ask_p])
                                qty = min(vol, remaining)
                                if qty > 0:
                                    orders.append(Order(sym, ask_p, qty))
                                    remaining -= qty
                                if remaining <= 0:
                                    break
                            else:
                                break
                    if remaining > 0 and best_bid is not None:
                        our_bid = best_bid + 1
                        if our_bid < fair_est + self.IPR_ENTRY_TAKE_EDGE:
                            orders.append(Order(sym, our_bid, remaining))
                elif target < pos:
                    remaining = pos - target
                    if od.buy_orders:
                        for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                            if bid_p >= fair_est - self.IPR_ENTRY_TAKE_EDGE:
                                vol = od.buy_orders[bid_p]
                                qty = min(vol, remaining)
                                if qty > 0:
                                    orders.append(Order(sym, bid_p, -qty))
                                    remaining -= qty
                                if remaining <= 0:
                                    break
                            else:
                                break
                    if remaining > 0 and best_ask is not None:
                        our_ask = best_ask - 1
                        if our_ask > fair_est - self.IPR_ENTRY_TAKE_EDGE:
                            orders.append(Order(sym, our_ask, -remaining))

                eff_cap = abs(target)
                if (regime == "UP" and pos > 0 and pos >= eff_cap - 5
                        and best_ask is not None):
                    our_ask = best_ask + self.IPR_SELL_QUOTE_OFFSET
                    sz = min(self.IPR_SELL_QUOTE_SIZE, pos)
                    if sz > 0 and our_ask > fair_est + 2:
                        orders.append(Order(sym, our_ask, -sz))
                elif (regime == "DOWN" and pos < 0 and pos <= -eff_cap + 5
                        and best_bid is not None):
                    our_bid = best_bid - self.IPR_SELL_QUOTE_OFFSET
                    sz = min(self.IPR_SELL_QUOTE_SIZE, -pos)
                    if sz > 0 and our_bid < fair_est - 2:
                        orders.append(Order(sym, our_bid, sz))

            if orders and not skip_throttle:
                orders = self._throttle(orders, pos, self.MAX_ENTRY_PER_TICK)
            if orders:
                result[sym] = orders

        return result, 0, self._save(ps)
