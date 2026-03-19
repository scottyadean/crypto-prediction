"""
crypto_predict.py
-----------------
Predicts which cryptocurrencies are most likely to go up
using a gradient boosting ML model trained on technical indicators.

Data sources:
  --source coingecko  (default) Free, no key needed. Add --api-key for real OHLC.
  --source tiingo     Real OHLCV candles. Requires --api-key or TIINGO_API_KEY env var.

Usage:
    python crypto_predict.py
    python crypto_predict.py --source tiingo --api-key YOUR_KEY --coins BTC ETH SOL
    python crypto_predict.py --source coingecko --api-key CG-xxx --top 5
    python crypto_predict.py --coins BTC ETH SOL --days 365
"""

import argparse
import os
import warnings
import time
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from tabulate import tabulate

# ─── Default coins ───────────────────────────────────────────────────────────
DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "ADA", "AVAX", "DOGE", "DOT", "LINK",
    "MATIC", "SHIB", "LTC", "UNI", "ATOM",
]

# CoinGecko needs full IDs (Tiingo uses the symbol directly as <ticker>usd)
SYMBOL_TO_CG_ID = {
    "BTC":   "bitcoin",      "ETH":   "ethereum",
    "SOL":   "solana",       "BNB":   "binancecoin",
    "XRP":   "ripple",       "ADA":   "cardano",
    "AVAX":  "avalanche-2",  "DOGE":  "dogecoin",
    "DOT":   "polkadot",     "LINK":  "chainlink",
    "MATIC": "matic-network","SHIB":  "shiba-inu",
    "LTC":   "litecoin",     "UNI":   "uniswap",
    "ATOM":  "cosmos",
}

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
TIINGO_BASE    = "https://api.tiingo.com/tiingo/crypto"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get(url, params, headers, timeout=20):
    """Thin requests wrapper with a clean error string on failure."""
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Tiingo ────────────────────────────────────────────────────────────────────
def fetch_tiingo(symbol: str, days: int, api_key: str) -> "pd.DataFrame | None":
    """
    Tiingo Crypto API — real daily OHLCV candles, no approximations.
    Correct endpoint: GET /tiingo/crypto/prices?tickers=btcusd&resampleFreq=1Day
    Docs: https://www.tiingo.com/documentation/crypto
    """
    ticker  = f"{symbol.lower()}usd"
    start   = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    headers = {"Content-Type": "application/json"}

    try:
        data = _get(
            f"{TIINGO_BASE}/prices",
            params={"tickers": ticker, "startDate": start, "resampleFreq": "1Day", "token": api_key},
            headers=headers,
        )
    except Exception as e:
        print(f"  \u26a0️  {symbol}: Tiingo fetch failed - {e}")
        return None

    if not data:
        return None

    # Response: list of {ticker, baseCurrency, priceData: [...]}
    price_data = None
    for entry in data:
        if entry.get("ticker", "").lower() == ticker:
            price_data = entry.get("priceData", [])
            break
    # Fallback: if response is a flat list of candles
    if price_data is None and data and "date" in data[0]:
        price_data = data

    if not price_data or len(price_data) < 60:
        return None

    df = pd.DataFrame(price_data)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates("date").set_index("date").sort_index()

    # Tiingo crypto uses camelCase — map to lowercase
    col_map = {"open": "open", "high": "high", "low": "low",
               "close": "close", "volume": "volume",
               "volumeNotional": "volume"}
    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        print(f"  \u26a0️  {symbol}: Tiingo response missing columns {missing}")
        return None

    time.sleep(0.2)
    return df[list(needed)].copy()


# ── CoinGecko ─────────────────────────────────────────────────────────────────
def fetch_coingecko(symbol: str, days: int, api_key: "str | None") -> "pd.DataFrame | None":
    """
    CoinGecko — two modes:
      With api_key  → /ohlc  (real candles) + /market_chart (volume)
      Without key   → /market_chart only (approx high/low from rolling close)
    """
    coin_id = SYMBOL_TO_CG_ID.get(symbol.upper())
    if not coin_id:
        print(f"  ⚠️  {symbol}: unknown CoinGecko ID — add it to SYMBOL_TO_CG_ID.")
        return None

    cg_days = "max" if days > 365 else days
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key

    if api_key:
        # Real OHLC candles
        try:
            raw = _get(f"{COINGECKO_BASE}/coins/{coin_id}/ohlc",
                       {"vs_currency": "usd", "days": cg_days}, headers)
        except Exception as e:
            print(f"  ⚠️  {symbol}: CoinGecko OHLC failed – {e}")
            return None

        if not raw or len(raw) < 60:
            return None

        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
        df = df.drop_duplicates("date").set_index("date").sort_index()

        try:
            time.sleep(0.15)
            vdata = _get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
                         {"vs_currency": "usd", "days": cg_days, "interval": "daily"}, headers)
            vol_df = pd.DataFrame(vdata["total_volumes"], columns=["ts", "volume"])
            vol_df["date"] = pd.to_datetime(vol_df["ts"], unit="ms").dt.normalize()
            df = df.join(vol_df.drop_duplicates("date").set_index("date")[["volume"]], how="left")
        except Exception:
            df["volume"] = 0

        df = df.fillna(0)
        sleep_secs = 0.15

    else:
        # Free path — market_chart only
        try:
            data = _get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
                        {"vs_currency": "usd", "days": cg_days, "interval": "daily"}, headers)
        except Exception as e:
            print(f"  ⚠️  {symbol}: CoinGecko fetch failed – {e}")
            return None

        if not data.get("prices") or len(data["prices"]) < 60:
            return None

        price_df = pd.DataFrame(data["prices"],        columns=["ts", "close"])
        vol_df   = pd.DataFrame(data["total_volumes"], columns=["ts", "volume"])
        for d in (price_df, vol_df):
            d["date"] = pd.to_datetime(d["ts"], unit="ms").dt.normalize()

        df = (
            price_df.drop_duplicates("date").set_index("date")[["close"]]
            .join(vol_df.drop_duplicates("date").set_index("date")[["volume"]], how="left")
            .sort_index().fillna(0)
        )
        df["high"] = df["close"].rolling(3, min_periods=1).max()
        df["low"]  = df["close"].rolling(3, min_periods=1).min()
        df["open"] = df["close"].shift(1).fillna(df["close"])
        sleep_secs = 1.2

    df = df.iloc[-days:]
    time.sleep(sleep_secs)
    return df[["open", "high", "low", "close", "volume"]].copy()


# ── Dispatcher ────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, days: int,
                source: str = "coingecko",
                api_key: "str | None" = None) -> "pd.DataFrame | None":
    if source == "tiingo":
        if not api_key:
            print(f"  ⚠️  Tiingo requires an API key. Pass --api-key or set TIINGO_API_KEY.")
            return None
        return fetch_tiingo(symbol, days, api_key)
    else:
        return fetch_coingecko(symbol, days, api_key)


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    v = df["volume"]

    for n in [1, 3, 7, 14, 30]:
        df[f"ret_{n}d"] = c.pct_change(n)

    for n in [7, 14, 21, 50]:
        df[f"sma_{n}"]          = c.rolling(n).mean()
        df[f"ema_{n}"]          = c.ewm(span=n, adjust=False).mean()
        df[f"price_vs_sma_{n}"] = c / df[f"sma_{n}"] - 1

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pct"] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)

    df["volatility_7d"]     = c.pct_change().rolling(7).std()
    df["volatility_30d"]    = c.pct_change().rolling(30).std()
    df["volume_change_7d"]  = v.pct_change(7)
    df["volume_sma_ratio"]  = v / (v.rolling(14).mean() + 1e-9)

    hl = df["high"] - df["low"]
    hc = (df["high"] - c.shift()).abs()
    lc = (df["low"]  - c.shift()).abs()
    df["atr_14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean() / (c + 1e-9)

    low14  = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-9)

    df["target"] = (c.shift(-7) > c).astype(int)
    return df.dropna()


FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_7d", "ret_14d", "ret_30d",
    "price_vs_sma_7", "price_vs_sma_14", "price_vs_sma_21", "price_vs_sma_50",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_pct", "volatility_7d", "volatility_30d",
    "volume_change_7d", "volume_sma_ratio",
    "atr_14", "stoch_k",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_predict(df: pd.DataFrame) -> dict:
    X, y    = df[FEATURE_COLS], df["target"]
    X_train = X.iloc[:-7];  y_train = y.iloc[:-7]

    scaler          = StandardScaler()
    X_tr_scaled     = scaler.fit_transform(X_train)
    X_latest_scaled = scaler.transform(X.iloc[[-1]])

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42,
    )
    cv_scores = []
    for tr, val in TimeSeriesSplit(n_splits=5).split(X_tr_scaled):
        model.fit(X_tr_scaled[tr], y_train.iloc[tr])
        cv_scores.append(accuracy_score(y_train.iloc[val], model.predict(X_tr_scaled[val])))

    model.fit(X_tr_scaled, y_train)
    prob_up = model.predict_proba(X_latest_scaled)[0][1]

    return {
        "prob_up":        prob_up,
        "cv_accuracy":    np.mean(cv_scores),
        "latest_price":   df["close"].iloc[-1],
        "ret_7d":         df["ret_7d"].iloc[-1],
        "rsi_14":         df["rsi_14"].iloc[-1],
        "volatility_30d": df["volatility_30d"].iloc[-1],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Crypto upside probability ranker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crypto_predict.py                                         # CoinGecko free
  python crypto_predict.py --source tiingo --api-key YOUR_KEY     # Tiingo real OHLCV
  python crypto_predict.py --source coingecko --api-key CG-xxx    # CoinGecko real OHLC
  python crypto_predict.py --coins BTC ETH SOL --top 3
        """,
    )
    parser.add_argument("--coins",   nargs="+", default=DEFAULT_COINS)
    parser.add_argument("--days",    type=int,  default=730,
                        help="Days of history to use (default: 730)")
    parser.add_argument("--top",     type=int,  default=None,
                        help="Show only top N results")
    parser.add_argument("--source",  type=str,  default="coingecko",
                        choices=["coingecko", "tiingo"],
                        help="Data source (default: coingecko)")
    parser.add_argument("--api-key", type=str,  default=None,
                        help="API key for the selected source. "
                             "Can also use COINGECKO_API_KEY or TIINGO_API_KEY env vars.")
    args = parser.parse_args()

    # Resolve API key: flag → env var for the selected source
    env_var = "TIINGO_API_KEY" if args.source == "tiingo" else "COINGECKO_API_KEY"
    print(f"  🔍 Using data source: {args.source} (API key from --api-key or {env_var} env var)")
    api_key = args.api_key or os.environ.get(env_var)

    # Human-readable source label
    if args.source == "tiingo":
        source_label = "Tiingo  (real OHLCV candles ✅)"
    elif api_key:
        source_label = "CoinGecko Demo/Paid  (real OHLC candles ✅)"
    else:
        source_label = "CoinGecko free  (approx high/low — use --api-key for real candles)"

    print(f"\n{'='*65}")
    print(f"  Crypto Upside Probability Ranker")
    print(f"  Data window : {args.days} days  |  Predicting: 7-day horizon")
    print(f"  Data source : {source_label}")
    print(f"{'='*65}\n")

    results = []
    for symbol in args.coins:
        print(f"  ⏳ Fetching {symbol}...", end="\r")
        df = fetch_ohlcv(symbol, days=args.days, source=args.source, api_key=api_key)
        if df is None:
            print(f"  ⚠️  {symbol}: no data returned, skipping.           ")
            continue
        df = add_features(df)
        if len(df) < 80:
            print(f"  ⚠️  {symbol}: too short after feature engineering, skipping.")
            continue
        try:
            stats = train_and_predict(df)
            results.append({"symbol": symbol, **stats})
            print(f"  ✅ {symbol}: prob_up={stats['prob_up']*100:.1f}%               ")
        except Exception as e:
            print(f"  ⚠️  {symbol}: model error – {e}")

    if not results:
        print("  No results — check connection, API key, or try --days 365.")
        return

    results.sort(key=lambda x: x["prob_up"], reverse=True)
    if args.top:
        results = results[: args.top]

    rows = []
    for rank, r in enumerate(results, 1):
        signal = ("🟢 BUY"   if r["prob_up"] >= 0.60 else
                  "🔴 AVOID" if r["prob_up"] <= 0.40 else
                  "🟡 HOLD")
        rows.append([
            rank, r["symbol"],
            f"${r['latest_price']:,.4f}",
            f"{r['prob_up']*100:.1f}%",
            f"{r['cv_accuracy']*100:.1f}%",
            f"{r['ret_7d']*100:+.1f}%",
            f"{r['rsi_14']:.1f}",
            f"{r['volatility_30d']*100:.2f}%",
            signal,
        ])

    headers = ["#", "Coin", "Price", "Prob Up", "CV Acc",
               "7d Ret", "RSI-14", "Vol-30d", "Signal"]
    print()
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print("""
  Prob Up  – Model's probability price is higher in 7 days
  CV Acc   – Time-series cross-validation directional accuracy
  Signal   – 🟢 ≥60%  |  🟡 40–60%  |  🔴 ≤40%

  ⚠️  NOT financial advice. Past patterns ≠ future returns.
""")


if __name__ == "__main__":
    main()