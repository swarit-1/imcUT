"""
IMC Prosperity 4 — Round 3 Trader (rewrite)

Replaces the prior static-anchor design that lost ~$80K in the live round.

Three core changes:
  1. Underlying fair value is now a pure EMA of microprice — no static anchor.
     Lets the MM track regime drift instead of fighting it.
  2. Voucher pricing fits an IV smile (parabolic in moneyness) from live
     market mids each tick. Options are quoted as relative-value vs the
     fitted curve; absolute vol level no longer matters.
  3. Risk uses a rolling z-score kill (not static-anchor distance) plus
     per-product PnL stops and an EOD unwind.

Tiers:
  1a. HYDROGEL_PACK MM around ema_hydrogel
  1b. VELVETFRUIT_EXTRACT MM around ema_velvet (= spot S for vouchers)
  2.  VEV_5000..5500 MM at theo from fitted smile
  3.  VEV_4000/4500 MM at theo from fitted smile (deep-ITM, ≈ S - K)
  4.  Net-delta hedge into VELVET book
  5.  Risk overlays — z-score kill, soft/hard PnL stops, EOD unwind
"""

import json
from math import log, sqrt, exp, erf, floor, ceil, isfinite
from typing import Dict, List, Tuple, Optional

from datamodel import Order, OrderDepth, TradingState, Symbol


# ---------------------------------------------------------------------------
# Products / position limits
# ---------------------------------------------------------------------------

HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"

VOUCHER_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEVS = {k: f"VEV_{k}" for k in VOUCHER_STRIKES}
VEV_TO_STRIKE = {v: k for k, v in VEVS.items()}

# 6000/6500 are clamped at min-tick (mid 0.5) — no edge available, skip.
ACTIVE_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500]
ACTIVE_VEVS = [VEVS[k] for k in ACTIVE_STRIKES]

POSITION_LIMITS = {HYDROGEL: 200, VELVET: 200}
for _v in VEVS.values():
    POSITION_LIMITS[_v] = 300


# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

# EMA half-life (in ticks) → α via α = 1 - 0.5**(1/halflife). Tuned to data:
#   HYDROGEL std≈25-37, spread≈16 → slow EMA tracks regime drift
#   VELVET   std≈14-17, spread≈5  → faster EMA, also serves as spot for options
EMA_ALPHA = {HYDROGEL: 0.01, VELVET: 0.05}

# MM edges (price units). EMA fair is too noisy for take decisions on the
# commodities, and the smile-fit residuals turn out to be noise rather than
# alpha. We disable taking everywhere (NO_TAKE_*) and rely on pure spread
# capture with wide edges.
NO_TAKE_COMMODITY = True
NO_TAKE_VEV = True

EDGE = {HYDROGEL: 4.0, VELVET: 2.0, "VEV": 2.0}
TAKE_EDGE = {HYDROGEL: 5.0, VELVET: 2.5, "VEV": 2.0}
MAX_SKEW = {HYDROGEL: 4.0, VELVET: 1.5, "VEV": 2.0}
QUOTE_SIZE = {HYDROGEL: 15, VELVET: 15, "VEV": 10}
# On tight-spread products where centered quoting would cross, open small
# two-sided BBO quotes when our fair is within this tick distance of book
# mid (model agrees with market). Earns the spread at the cost of some
# adverse selection. Size is capped aggressively below.
TIGHT_SPREAD_MAX_MODEL_DIFF = 1.0
TIGHT_SPREAD_SIZE = 5

# Delta hedging (in VELVET units)
DELTA_BAND = 100
HEDGE_PER_TICK = 30

# Z-score kill
KILL_Z = 5.0
WINDOW_LEN = 200

# PnL stops (per product)
SOFT_STOP = -3000.0
HARD_STOP = -8000.0

# EOD unwind starts at this timestamp
EOD_UNWIND_TS = 950_000

# Smile fit
SMILE_MIN_POINTS = 4
FALLBACK_IV = 0.0127       # used until first valid smile fit
MIN_VEV_MID_FOR_FIT = 1.5  # ignore sub-tick noise

# TTE handling
LIVE_DAY = None            # set to int (e.g. 3) to skip auto-detect
DAYS_TO_EXPIRY_AT_DAY_1 = 7
TICKS_PER_DAY = 1_000_000.0


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

_INV_SQRT_2 = 1.0 / sqrt(2.0)
_INV_SQRT_2PI = 1.0 / sqrt(2.0 * 3.141592653589793)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x * _INV_SQRT_2))


def norm_pdf(x: float) -> float:
    return _INV_SQRT_2PI * exp(-0.5 * x * x)


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return max(S - K, 0.0)
    sqT = sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return 1.0 if S > K else 0.0
    sqT = sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    return norm_cdf(d1)


def bs_call_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return 0.0
    sqT = sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    return S * norm_pdf(d1) * sqT


def bs_implied_vol(price: float, S: float, K: float, T: float) -> Optional[float]:
    """Newton's method. Returns None if it can't converge."""
    if price <= 0.0 or S <= 0.0 or T <= 0.0:
        return None
    intrinsic = max(S - K, 0.0)
    if price < intrinsic - 0.5:
        return None  # below intrinsic — likely stale book
    sigma = 0.013
    for _ in range(40):
        p = bs_call_price(S, K, T, sigma)
        v = bs_call_vega(S, K, T, sigma)
        if v < 1e-9:
            break
        diff = p - price
        if abs(diff) < 1e-5:
            return sigma
        sigma -= diff / v
        if sigma <= 1e-6:
            sigma = 1e-6
        if sigma > 5.0:
            sigma = 5.0
    if 1e-5 < sigma < 5.0 and isfinite(sigma):
        return sigma
    return None


def fit_quadratic_wls(xs: List[float], ys: List[float], ws: List[float]) -> Optional[Tuple[float, float, float]]:
    """Weighted least squares fit y = a + b*x + c*x^2 via 3x3 normal equations."""
    n = len(xs)
    if n < 3 or n != len(ys) or n != len(ws):
        return None

    s0 = s1 = s2 = s3 = s4 = 0.0
    t0 = t1 = t2 = 0.0
    for i in range(n):
        w = ws[i]
        x = xs[i]
        y = ys[i]
        x2 = x * x
        s0 += w
        s1 += w * x
        s2 += w * x2
        s3 += w * x2 * x
        s4 += w * x2 * x2
        t0 += w * y
        t1 += w * x * y
        t2 += w * x2 * y

    # Solve  | s0 s1 s2 | | a |   | t0 |
    #        | s1 s2 s3 | | b | = | t1 |
    #        | s2 s3 s4 | | c |   | t2 |
    M = [[s0, s1, s2, t0],
         [s1, s2, s3, t1],
         [s2, s3, s4, t2]]
    # Gaussian elimination with partial pivot
    for col in range(3):
        piv = col
        for r in range(col + 1, 3):
            if abs(M[r][col]) > abs(M[piv][col]):
                piv = r
        if abs(M[piv][col]) < 1e-12:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        inv = 1.0 / M[col][col]
        for r in range(col + 1, 3):
            f = M[r][col] * inv
            for k in range(col, 4):
                M[r][k] -= f * M[col][k]
    # Back-substitute
    sol = [0.0, 0.0, 0.0]
    for r in range(2, -1, -1):
        x = M[r][3]
        for k in range(r + 1, 3):
            x -= M[r][k] * sol[k]
        sol[r] = x / M[r][r]
    a, b, c = sol
    if not (isfinite(a) and isfinite(b) and isfinite(c)):
        return None
    return a, b, c


def smile_iv(smile: Optional[Tuple[float, float, float]], m: float) -> float:
    if smile is None:
        return FALLBACK_IV
    a, b, c = smile
    iv = a + b * m + c * m * m
    if not isfinite(iv) or iv < 1e-4 or iv > 5.0:
        return FALLBACK_IV
    return iv


# ---------------------------------------------------------------------------
# Order book helpers
# ---------------------------------------------------------------------------

def best_bid(depth: OrderDepth):
    if not depth.buy_orders:
        return None, 0
    p = max(depth.buy_orders.keys())
    return p, depth.buy_orders[p]


def best_ask(depth: OrderDepth):
    if not depth.sell_orders:
        return None, 0
    p = min(depth.sell_orders.keys())
    return p, -depth.sell_orders[p]


def mid_price(depth: OrderDepth):
    bp, _ = best_bid(depth)
    ap, _ = best_ask(depth)
    if bp is None or ap is None:
        return None
    return 0.5 * (bp + ap)


def microprice(depth: OrderDepth):
    bp, bv = best_bid(depth)
    ap, av = best_ask(depth)
    if bp is None or ap is None:
        return None
    if bv + av == 0:
        return 0.5 * (bp + ap)
    return (bp * av + ap * bv) / (bv + av)


# ---------------------------------------------------------------------------
# Generic market-making routine
# ---------------------------------------------------------------------------

def make_market(
    symbol: str,
    fair: float,
    depth: OrderDepth,
    position: int,
    edge: float,
    take_edge: float,
    quote_size_cap: int,
    skew_shift: float,
    eod_unwind: bool,
    take_only: bool = False,
    no_take: bool = False,
    min_price: int = 1,
) -> List[Order]:
    """Generate take + make orders for one product.

    `take_only` skips the make phase (used during EOD unwind alongside
    explicit flatten orders, or by the soft-stop overlay which keeps
    only passive quotes — see callers).
    `no_take` skips the take phase (used in soft-stop mode).
    """
    orders: List[Order] = []
    limit = POSITION_LIMITS[symbol]

    pos_after_buy = position
    pos_after_sell = position

    if not no_take:
        for ask in sorted(depth.sell_orders.keys()):
            if ask > fair - take_edge:
                break
            avail = -depth.sell_orders[ask]
            room = limit - pos_after_buy
            if room <= 0:
                break
            qty = min(avail, room)
            if qty > 0:
                orders.append(Order(symbol, ask, qty))
                pos_after_buy += qty

        for bid in sorted(depth.buy_orders.keys(), reverse=True):
            if bid < fair + take_edge:
                break
            avail = depth.buy_orders[bid]
            room = pos_after_sell + limit
            if room <= 0:
                break
            qty = min(avail, room)
            if qty > 0:
                orders.append(Order(symbol, bid, -qty))
                pos_after_sell -= qty

    if take_only:
        return orders

    bp, _ = best_bid(depth)
    ap, _ = best_ask(depth)
    if bp is None or ap is None:
        return orders

    book_mid = 0.5 * (bp + ap)
    quote_center = book_mid - skew_shift

    quote_bid = int(floor(quote_center - edge))
    quote_ask = int(ceil(quote_center + edge))

    if quote_bid <= bp:
        quote_bid = bp + 1
    if quote_ask >= ap:
        quote_ask = ap - 1

    quote_bid = max(quote_bid, min_price)
    quote_ask = max(quote_ask, min_price + 1)

    # Tight-spread fallback: penny-improvement crossed, so a centered
    # two-sided quote doesn't fit in the spread. Three sub-cases:
    #  (a) Inventory to offload: single-side at BBO on the reducing side.
    #  (b) Flat inventory, model close to book mid: small two-sided quote
    #      at BBO to earn the spread (accepts some adverse selection, so
    #      we cap size at TIGHT_SPREAD_SIZE and require model-book agreement).
    #  (c) Model disagrees with book mid by more than the threshold: stay out.
    tight_spread = quote_bid >= quote_ask or quote_bid >= ap or quote_ask <= bp
    if tight_spread:
        if eod_unwind:
            if position > 0:
                ask_room = pos_after_sell + limit
                sz = min(quote_size_cap, ask_room, position)
                if sz > 0:
                    orders.append(Order(symbol, ap, -sz))
            elif position < 0:
                bid_room = limit - pos_after_buy
                sz = min(quote_size_cap, bid_room, -position)
                if sz > 0:
                    orders.append(Order(symbol, bp, sz))
            return orders
        if position > 0:
            ask_room = pos_after_sell + limit
            sz = min(quote_size_cap, ask_room, position)
            if sz > 0:
                orders.append(Order(symbol, ap, -sz))
            return orders
        if position < 0:
            bid_room = limit - pos_after_buy
            sz = min(quote_size_cap, bid_room, -position)
            if sz > 0:
                orders.append(Order(symbol, bp, sz))
            return orders
        # position == 0: small two-sided quote if model agrees with book
        if abs(fair - book_mid) <= TIGHT_SPREAD_MAX_MODEL_DIFF:
            bid_room = limit - pos_after_buy
            ask_room = pos_after_sell + limit
            bid_size = min(TIGHT_SPREAD_SIZE, bid_room)
            ask_size = min(TIGHT_SPREAD_SIZE, ask_room)
            if bid_size > 0:
                orders.append(Order(symbol, bp, bid_size))
            if ask_size > 0:
                orders.append(Order(symbol, ap, -ask_size))
        return orders

    bid_room = limit - pos_after_buy
    ask_room = pos_after_sell + limit
    bid_size = min(quote_size_cap, bid_room)
    ask_size = min(quote_size_cap, ask_room)

    if eod_unwind:
        if position > 0:
            bid_size = 0
        elif position < 0:
            ask_size = 0
        else:
            bid_size = ask_size = 0

    if bid_size > 0:
        orders.append(Order(symbol, quote_bid, bid_size))
    if ask_size > 0:
        orders.append(Order(symbol, quote_ask, -ask_size))

    return orders


def skew_amount(position: int, limit: int, max_skew: float) -> float:
    if limit <= 0:
        return 0.0
    return max_skew * position / limit


def flatten_orders(symbol: str, depth: OrderDepth, position: int, chunk: int = 25) -> List[Order]:
    """Cross book in `chunk` units to flatten `position`. One chunk per call."""
    if position == 0:
        return []
    if position > 0:
        bp, bv = best_bid(depth)
        if bp is None:
            return []
        qty = -min(chunk, position, bv if bv > 0 else chunk)
        return [Order(symbol, bp, qty)] if qty < 0 else []
    else:
        ap, av = best_ask(depth)
        if ap is None:
            return []
        qty = min(chunk, -position, av if av > 0 else chunk)
        return [Order(symbol, ap, qty)] if qty > 0 else []


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------

class Trader:
    def __init__(self):
        self._defaults = {
            "ticks": 0,
            "ema": {},          # sym -> float
            "mid_window": {},   # sym -> [floats]
            "smile": None,      # [a, b, c] or None
            "cash": {},         # sym -> float (realized)
            "trade_seen_ts": {},  # sym -> last-processed trade timestamp
            "halted": {},       # sym -> bool
            "live_day": None,   # int or None
        }

    def _load_state(self, traderData: str) -> dict:
        if not traderData:
            return json.loads(json.dumps(self._defaults))  # deep copy
        try:
            s = json.loads(traderData)
            for k, v in self._defaults.items():
                s.setdefault(k, v if not isinstance(v, (dict, list)) else json.loads(json.dumps(v)))
            return s
        except Exception:
            return json.loads(json.dumps(self._defaults))

    @staticmethod
    def _save_state(s: dict) -> str:
        return json.dumps(s, separators=(",", ":"))

    # -----------------------------------------------------------------
    # Cash / PnL accounting
    # -----------------------------------------------------------------

    def _update_cash(self, state: TradingState, s: dict) -> None:
        """Apply own_trades since last call to running cash totals."""
        cash: Dict[str, float] = s["cash"]
        seen: Dict[str, int] = s["trade_seen_ts"]
        for sym, trades in (state.own_trades or {}).items():
            last_ts = seen.get(sym, -1)
            new_last = last_ts
            for t in trades:
                ts = getattr(t, "timestamp", 0) or 0
                if ts <= last_ts:
                    continue
                qty = abs(t.quantity)
                price = float(t.price)
                # IMC convention: SUBMISSION = us
                if getattr(t, "buyer", "") == "SUBMISSION":
                    cash[sym] = cash.get(sym, 0.0) - price * qty
                elif getattr(t, "seller", "") == "SUBMISSION":
                    cash[sym] = cash.get(sym, 0.0) + price * qty
                if ts > new_last:
                    new_last = ts
            seen[sym] = new_last

    def _product_pnl(self, sym: str, s: dict, position: int, mark: Optional[float]) -> float:
        cash = s["cash"].get(sym, 0.0)
        if mark is None:
            return cash
        return cash + position * mark

    # -----------------------------------------------------------------
    # Fair value (EMA) + rolling window for z-score
    # -----------------------------------------------------------------

    def _update_fair_window(self, sym: str, depth: OrderDepth, s: dict) -> Optional[float]:
        m = microprice(depth) if depth else None
        if m is None:
            m = mid_price(depth) if depth else None
        if m is None:
            return s["ema"].get(sym)

        alpha = EMA_ALPHA.get(sym, 0.05)
        prev = s["ema"].get(sym)
        ema = m if prev is None else alpha * m + (1 - alpha) * prev
        s["ema"][sym] = ema

        win = s["mid_window"].get(sym)
        if win is None:
            win = []
            s["mid_window"][sym] = win
        win.append(m)
        if len(win) > WINDOW_LEN:
            del win[: len(win) - WINDOW_LEN]
        return ema

    @staticmethod
    def _z_score(win: List[float], current: Optional[float]) -> float:
        if current is None or not win or len(win) < 20:
            return 0.0
        n = len(win)
        mean = sum(win) / n
        var = sum((x - mean) ** 2 for x in win) / n
        if var <= 0:
            return 0.0
        return (current - mean) / sqrt(var)

    # -----------------------------------------------------------------
    # TTE / smile
    # -----------------------------------------------------------------

    def _resolve_live_day(self, s: dict, S: float, depths: Dict[str, OrderDepth]) -> int:
        """Pick LIVE_DAY by minimizing smile residuals across candidate days."""
        if LIVE_DAY is not None:
            return LIVE_DAY
        cached = s.get("live_day")
        if cached is not None:
            return cached

        candidates = [2, 3, 4, 5, 6, 7]
        best_day = 3
        best_score = float("inf")
        for d in candidates:
            tte = max(0.05, (8 - d))
            pts = self._collect_smile_points(S, tte, depths)
            if len(pts) < 4:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ws = [p[2] for p in pts]
            fit = fit_quadratic_wls(xs, ys, ws)
            if fit is None:
                continue
            a, b, c = fit
            ss = 0.0
            for x, y, w in pts:
                pred = a + b * x + c * x * x
                ss += w * (y - pred) ** 2
            if ss < best_score:
                best_score = ss
                best_day = d
        s["live_day"] = best_day
        return best_day

    @staticmethod
    def _collect_smile_points(S: float, T: float, depths: Dict[str, OrderDepth]) -> List[Tuple[float, float, float]]:
        """Return list of (moneyness m, IV σ, weight) for fittable strikes."""
        pts = []
        sqT = sqrt(T) if T > 0 else 1.0
        for K in ACTIVE_STRIKES:
            sym = VEVS[K]
            d = depths.get(sym)
            if d is None:
                continue
            bp, bv = best_bid(d)
            ap, av = best_ask(d)
            if bp is None or ap is None:
                continue
            mid = 0.5 * (bp + ap)
            if mid < MIN_VEV_MID_FOR_FIT:
                continue
            iv = bs_implied_vol(mid, S, K, T)
            if iv is None or iv < 1e-4 or iv > 1.0:
                continue
            m = log(K / S) / sqT
            spread = ap - bp
            depth_score = min(bv, av)
            w = max(1.0, depth_score) / max(1.0, spread)  # tight + deep = high weight
            pts.append((m, iv, w))
        return pts

    # -----------------------------------------------------------------
    # Main entry
    # -----------------------------------------------------------------

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        s = self._load_state(state.traderData)
        s["ticks"] += 1
        ts = state.timestamp
        eod = ts >= EOD_UNWIND_TS
        result: Dict[Symbol, List[Order]] = {}

        positions = state.position or {}
        depths = state.order_depths or {}

        # Update cash from own_trades reported this tick
        self._update_cash(state, s)

        # Update EMAs and rolling windows for HYDROGEL + VELVET (and VEVs for z-kill)
        ema_h = self._update_fair_window(HYDROGEL, depths.get(HYDROGEL), s) if HYDROGEL in depths else None
        ema_v = self._update_fair_window(VELVET, depths.get(VELVET), s) if VELVET in depths else None
        for v in ACTIVE_VEVS:
            if v in depths:
                self._update_fair_window(v, depths[v], s)

        S = ema_v  # spot for option pricing

        # Resolve LIVE_DAY (cached after first successful detection)
        if S is not None:
            live_day = self._resolve_live_day(s, S, depths)
        else:
            live_day = LIVE_DAY if LIVE_DAY is not None else 3
        TTE = max(1e-6, (8 - live_day) - ts / TICKS_PER_DAY)

        # Fit smile from this tick's market mids
        smile = None
        if S is not None and TTE > 0:
            pts = self._collect_smile_points(S, TTE, depths)
            if len(pts) >= SMILE_MIN_POINTS:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ws = [p[2] for p in pts]
                fit = fit_quadratic_wls(xs, ys, ws)
                if fit is not None:
                    a, b, c = fit
                    if 0.0001 < a < 1.0 and abs(b) < 5.0 and abs(c) < 5.0:
                        smile = (a, b, c)
                        s["smile"] = list(smile)
            if smile is None and s.get("smile"):
                smile = tuple(s["smile"])

        # Helpers for risk overlay
        def stops_for(sym: str) -> Tuple[bool, bool]:
            """Return (no_take, take_only) given product PnL."""
            mark = s["ema"].get(sym)
            pnl = self._product_pnl(sym, s, positions.get(sym, 0), mark)
            if s["halted"].get(sym):
                return True, True  # caller will use flatten_orders instead
            if pnl < HARD_STOP:
                s["halted"][sym] = True
                return True, True
            if pnl < SOFT_STOP:
                return True, False  # passive only, no taking
            return False, False

        def z_kill(sym: str, mid_now: Optional[float]) -> bool:
            win = s["mid_window"].get(sym, [])
            return abs(self._z_score(win, mid_now)) > KILL_Z

        # ---------------- Tier 1a: HYDROGEL_PACK MM ----------------
        if HYDROGEL in depths and ema_h is not None:
            hpos = positions.get(HYDROGEL, 0)
            hmid = mid_price(depths[HYDROGEL])
            no_take, take_only = stops_for(HYDROGEL)
            if s["halted"].get(HYDROGEL):
                result[HYDROGEL] = flatten_orders(HYDROGEL, depths[HYDROGEL], hpos)
            elif z_kill(HYDROGEL, hmid):
                pass  # skip MM this tick
            else:
                hskew = skew_amount(hpos, POSITION_LIMITS[HYDROGEL], MAX_SKEW[HYDROGEL])
                edge_h = EDGE[HYDROGEL] * (2.0 if no_take else 1.0)
                result[HYDROGEL] = make_market(
                    HYDROGEL, ema_h, depths[HYDROGEL], hpos,
                    edge_h, TAKE_EDGE[HYDROGEL],
                    QUOTE_SIZE[HYDROGEL], hskew, eod,
                    no_take=no_take or NO_TAKE_COMMODITY, take_only=take_only,
                )

        # ---------------- Tier 1b: VELVET MM ----------------
        if VELVET in depths and ema_v is not None:
            vpos = positions.get(VELVET, 0)
            vmid = mid_price(depths[VELVET])
            no_take, take_only = stops_for(VELVET)
            if s["halted"].get(VELVET):
                result[VELVET] = flatten_orders(VELVET, depths[VELVET], vpos)
            elif z_kill(VELVET, vmid):
                pass
            else:
                vskew = skew_amount(vpos, POSITION_LIMITS[VELVET], MAX_SKEW[VELVET])
                edge_v = EDGE[VELVET] * (2.0 if no_take else 1.0)
                result[VELVET] = make_market(
                    VELVET, ema_v, depths[VELVET], vpos,
                    edge_v, TAKE_EDGE[VELVET],
                    QUOTE_SIZE[VELVET], vskew, eod,
                    no_take=no_take or NO_TAKE_COMMODITY, take_only=take_only,
                )

        # ---------------- Tier 2-3: VEV vouchers ----------------
        net_delta = 0.0
        if S is not None and S > 0 and TTE > 0:
            sqT = sqrt(TTE)
            for K in ACTIVE_STRIKES:
                sym = VEVS[K]
                if sym not in depths:
                    continue
                pos = positions.get(sym, 0)
                m_K = log(K / S) / sqT
                sigma = smile_iv(smile, m_K)
                theo = bs_call_price(S, K, TTE, sigma)
                delta_K = bs_call_delta(S, K, TTE, sigma)
                net_delta += pos * delta_K

                vmid = mid_price(depths[sym])

                if s["halted"].get(sym):
                    result[sym] = flatten_orders(sym, depths[sym], pos, chunk=20)
                    continue
                if z_kill(sym, vmid):
                    continue
                if theo < 1.0:
                    # Sub-tick fair — quoting becomes noise
                    continue

                no_take, take_only = stops_for(sym)
                edge_v = EDGE["VEV"] * (2.0 if no_take else 1.0)
                vskew = skew_amount(pos, POSITION_LIMITS[sym], MAX_SKEW["VEV"])
                result[sym] = make_market(
                    sym, theo, depths[sym], pos,
                    edge_v, TAKE_EDGE["VEV"],
                    QUOTE_SIZE["VEV"], vskew, eod,
                    no_take=no_take or NO_TAKE_VEV, take_only=take_only,
                )

        # ---------------- Tier 4: Net-delta hedge ----------------
        velvet_depth = depths.get(VELVET)
        if velvet_depth is not None and not s["halted"].get(VELVET):
            net_delta_total = positions.get(VELVET, 0) + net_delta
            if abs(net_delta_total) > DELTA_BAND and not eod:
                existing = result.get(VELVET, [])
                vpos = positions.get(VELVET, 0)
                limit = POSITION_LIMITS[VELVET]
                # Don't double-count any inventory we're already moving via existing orders
                pending = sum(o.quantity for o in existing)
                room_buy = limit - vpos - max(0, pending)
                room_sell = vpos + limit + min(0, pending)
                if net_delta_total > 0:
                    bp, bv = best_bid(velvet_depth)
                    if bp is not None:
                        target = int(round(net_delta_total - DELTA_BAND / 2))
                        qty = -min(HEDGE_PER_TICK, target, room_sell, bv if bv > 0 else HEDGE_PER_TICK)
                        if qty < 0:
                            existing.append(Order(VELVET, bp, qty))
                else:
                    ap, av = best_ask(velvet_depth)
                    if ap is not None:
                        target = int(round(-net_delta_total - DELTA_BAND / 2))
                        qty = min(HEDGE_PER_TICK, target, room_buy, av if av > 0 else HEDGE_PER_TICK)
                        if qty > 0:
                            existing.append(Order(VELVET, ap, qty))
                result[VELVET] = existing

        # ---------------- EOD unwind: flatten remaining inventory ----------------
        if eod:
            for sym, depth in depths.items():
                pos = positions.get(sym, 0)
                if pos == 0:
                    continue
                # Override any existing MM with a flatten
                result[sym] = flatten_orders(sym, depth, pos, chunk=25)

        # ---------------- Final position-limit clip ----------------
        for sym, orders in list(result.items()):
            limit = POSITION_LIMITS.get(sym, 0)
            pos = positions.get(sym, 0)
            buy_room = limit - pos
            sell_room = pos + limit
            cleaned: List[Order] = []
            for o in orders:
                if o.quantity > 0:
                    q = min(o.quantity, buy_room)
                    if q > 0:
                        cleaned.append(Order(sym, o.price, q))
                        buy_room -= q
                elif o.quantity < 0:
                    q = max(o.quantity, -sell_room)
                    if q < 0:
                        cleaned.append(Order(sym, o.price, q))
                        sell_room += q
            result[sym] = cleaned

        return result, 0, self._save_state(s)
