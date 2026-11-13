# polymarket-nvidia-ceo-forecast-01-entry

[![make.fun · polymarket-nvidia-ceo-forecast-01](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01)

Forecast the Yes price of every trading market of Polymarket's "Next CEO of
NVIDIA?" event, minute by minute, 4 hours ahead — five quantiles per minute.

The entry program is `main.py` (Python 3.13, standard library only). It reads
one case file from `/input` and writes one forecast file to `/output`, offline
and deterministic: two runs write identical bytes.

Method: the median forecast is the last price (prices are a near-martingale
with rare, mostly transient jumps). The spread is per-market: empirical
quantiles of h-step price changes over the last 24 hours at horizons 1, 2,
4, 8, 16, 32, 64, 128, 240 minutes, shrunk 50% toward zero and interpolated
in log-h, blended 50/50 with a Normal spread from the market's robust
1-minute volatility (1.4826 * MAD, floored 0.0008, capped 0.02), widening
like sqrt(h).

- Competition: make.fun `polymarket-nvidia-ceo-forecast-01` — $250 in USDC,
  100 places paid (1st $40)
- Practice backtest (six practice cases vs Polymarket's own history):
  loss ~0.299 vs house baseline 0.363589
- License: MIT
