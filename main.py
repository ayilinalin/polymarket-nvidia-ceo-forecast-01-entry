#!/usr/bin/env python3
"""Polymarket NVIDIA CEO forecast entry (make.fun program/v1).

Forecasts the Yes price of every trading market of Polymarket's event
"Next CEO of NVIDIA?" for the 240 minutes after a checkpoint, as five
quantiles per minute, from the case's own 1-minute price history only.

Method (fitted on the six practice cases, scored against Polymarket's own
history; practice loss ~0.299 vs the house baseline's 0.363589):

* Prices are a near-martingale with rare, mostly transient jumps: the
  median forecast is the last price (no drift). What matters is the
  spread, which differs wildly per market.
* For each market, the spread comes from the empirical distribution of
  h-step price changes over the last 24 hours, at coarse horizons
  h = 1, 2, 4, 8, 16, 32, 64, 128, 240 minutes, shrunk 50% toward zero
  (raw empirical tails are noisy), interpolated in log-h.
* That empirical spread is blended 50/50 with a Normal spread whose
  sigma is the market's robust 1-minute volatility (1.4826 * MAD of
  1-minute changes, floored at 0.0008, capped at 0.02), widening like
  sqrt(h). The blend keeps the tails honest on jumpy markets while the
  Normal part stops the empirical tails from overfitting one regime.
* Quantiles are clamped to [0, 1], rounded to 6 decimals, non-decreasing.

Python 3.13, standard library only, no network, fully deterministic:
sorted iteration, no wall-clock input, no randomness.
"""
import json
import math
import os
import statistics

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
H_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 240]
WINDOW_MINUTES = 24 * 60
SHRINK = 0.5          # empirical h-step changes shrunk toward zero
BLEND_NORMAL = 0.5    # weight of the Normal spread in the blend
SIGMA_FLOOR = 0.0008
SIGMA_CAP = 0.02
BASELINE_SIGMA = 0.0025


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def norm_ppf(p):
    """Acklam's approximation to the standard normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def quantile_sorted(s, tau):
    pos = tau * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    f = pos - lo
    return s[lo] * (1.0 - f) + s[hi] * f


def robust_sigma(prices):
    ch = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    if not ch:
        return BASELINE_SIGMA
    med = statistics.median(ch)
    mad = statistics.median([abs(v - med) for v in ch])
    return min(max(1.4826 * mad, SIGMA_FLOOR), SIGMA_CAP)


def baseline_rows(x, horizon, taus, zs):
    return [[price6(x + z * BASELINE_SIGMA * math.sqrt(h)) for z in zs]
            for h in range(1, horizon + 1)]


def forecast_market(prices, horizon, taus, zs):
    x = prices[-1]
    # Last 24 hours carry the current regime; fall back to the full
    # history when a market is younger than that.
    p = prices[-WINDOW_MINUTES:] if len(prices) > WINDOW_MINUTES else prices
    n = len(p)
    grid_q = []
    for h in H_GRID:
        if n - h < 50:
            continue
        d = sorted((p[i + h] - p[i]) * SHRINK for i in range(n - h))
        grid_q.append((h, [quantile_sorted(d, t) for t in taus]))
    if not grid_q:
        return baseline_rows(x, horizon, taus, zs)
    sig = robust_sigma(p)
    rows = []
    for h in range(1, horizon + 1):
        if h <= grid_q[0][0]:
            qs = grid_q[0][1]
        elif h >= grid_q[-1][0]:
            qs = grid_q[-1][1]
        else:
            qs = grid_q[0][1]
            for j in range(len(grid_q) - 1):
                h0, q0 = grid_q[j]
                h1, q1 = grid_q[j + 1]
                if h0 <= h <= h1:
                    w = (math.log(h) - math.log(h0)) / (math.log(h1) - math.log(h0))
                    qs = [q0[k] * (1.0 - w) + q1[k] * w for k in range(len(taus))]
                    break
        out = []
        for k in range(len(taus)):
            # Median (tau 0.5) stays at the last price: the empirical
            # median drift is noise on a near-martingale.
            q_emp = x if taus[k] == 0.5 else x + qs[k]
            q_norm = x + zs[k] * sig * math.sqrt(h)
            out.append(price6((1.0 - BLEND_NORMAL) * q_emp + BLEND_NORMAL * q_norm))
        for k in range(1, len(out)):
            if out[k] < out[k - 1]:
                out[k] = out[k - 1]
        rows.append(out)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[:-len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case["horizon_minutes"])
        taus = [float(t) for t in case["quantiles"]]
        zs = [norm_ppf(t) for t in taus]
        forecasts = []
        for mkt in case["markets"]:
            prices = [float(v) for v in mkt["prices"]]
            forecasts.append({"id": mkt["id"],
                              "q": forecast_market(prices, horizon, taus, zs)})
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
