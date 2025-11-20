import os
import pandas as pd
import numpy as np
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
fred = Fred(os.getenv("FRED_API_KEY"))

# Define all raw FRED series needed for final FIS system
FRED_SERIES = {
    # Rates, Spreads, Inflation
    "DGS10": {"desc": "10Y Treasury Yield", "freq": "D", "window": 120},
    "DGS2":  {"desc": "2Y Treasury Yield",  "freq": "D", "window": 120},
    "FEDFUNDS": {"desc": "Fed Funds Rate", "freq": "M", "window": 60},
    "CPIAUCSL": {"desc": "CPI Level", "freq": "M", "window": 60},

    # Unemployment (for SP5MV)
    "UNRATE": {"desc": "Unemployment Rate", "freq": "M", "window": 60},

    # Credit spreads
    "BAMLH0A0HYM2": {"desc": "High-Yield Spread", "freq": "D", "window": 60},

    # Growth and Production
    "A191RL1Q225SBEA": {"desc": "Real GDP", "freq": "Q", "window": 60},
    "INDPRO": {"desc": "Industrial Production", "freq": "M", "window": 36},
    "PAYEMS": {"desc": "Payroll Employment", "freq": "M", "window": 36},
    "CP": {"desc": "Corporate Profits", "freq": "Q", "window": 60},

    # Sentiment, Volatility, Liquidity
    "UMCSENT": {"desc": "Consumer Sentiment", "freq": "M", "window": 36},
    "VIXCLS": {"desc": "Volatility Index", "freq": "D", "window": 36}, # used to be VXO, but that was discontinued
    "M2SL": {"desc": "Money Supply M2", "freq": "M", "window": 36},
    "USEPUINDXD": {"desc": "Policy Uncertainty", "freq": "M", "window": 36},

    # Capex proxy
    "PNFIC1": {"desc": "Private Nonresidential Fixed Investment", "freq": "Q", "window": 60},
}

START_DATE = "2000-01-01"
END_DATE   = "2023-12-31"
DATA_DIR   = "fred"
os.makedirs(DATA_DIR, exist_ok=True)



# Fetch a series and convert to monthly
def fetch_and_resample(code, meta):
    print(f"Fetching {code}: {meta['desc']}")
    series = fred.get_series(code, observation_start=START_DATE, observation_end=END_DATE)

    if series is None or series.empty:
        print(f"WARNING: {code} returned empty.")
        return pd.Series(dtype=float)

    if meta["freq"] == "D":
        s = series.resample("ME").last()
    elif meta["freq"] == "Q":
        s = series.resample("QE").last().resample("ME").ffill()
    else:  # Monthly
        s = series.resample("ME").last()

    s.name = code
    return s



# Rolling Z-score with per-series lookback
def rolling_z(series: pd.Series, window: int):
    """Stable rolling z-score."""
    roll = series.rolling(window)
    return (series - roll.mean()) / roll.std()


# Build raw dataset
def build_raw_dataset():
    frames = []
    for code, meta in FRED_SERIES.items():
        s = fetch_and_resample(code, meta) # resample to monthly
        frames.append(s)

    df = pd.concat(frames, axis=1)
    df = df.dropna(how="all")
    df.to_csv(f"{DATA_DIR}/raw_fred_data.csv")
    print(f"Saved raw_fred_data.csv with shape {df.shape}")
    return df

# Build zscored and derived macro dataset
def build_final_zscored(df: pd.DataFrame):
    dfz = pd.DataFrame(index=df.index)

    # Base z-scores
    for code, meta in FRED_SERIES.items():
        if code not in df: continue
        w = meta["window"]
        dfz[f"{code}_z"] = rolling_z(df[code], w)

    # Derived Series
    # we have to calculate YC raw then rolling z it (5 years)
    dfz["YC_raw"] = df["DGS10"] - df["DGS2"]
    dfz["YC_z"] = rolling_z(dfz["YC_raw"], 60)

    # CPI YoY for real FF rate
    cpi_yoy = df["CPIAUCSL"].pct_change(12) * 100
    dfz["CPI_YoY"] = cpi_yoy
    dfz["CPI_YoY_z"] = rolling_z(cpi_yoy, 60)

    # Real Fed Funds Rate (raw)
    dfz["RFF_raw"] = df["FEDFUNDS"] - cpi_yoy
    dfz["RFF_z"] = rolling_z(dfz["RFF_raw"], 60)

    # Real Unemployment
    dfz["UNRATE_z"] = rolling_z(df["UNRATE"], 60)

    #  Clip all series
    dfz = dfz.clip(-2.5, 2.5)

    # Final clean
    dfz = dfz.dropna(how="all")
    dfz.to_csv(f"{DATA_DIR}/zscored_clipped.csv")
    print(f"Saved final zscored_clipped.csv with shape {dfz.shape}")

    return dfz


# main
if __name__ == "__main__":
    print("\n=== Building FRED Database ===\n")
    raw = build_raw_dataset()
    final_df = build_final_zscored(raw)

    print("\nPreview of final dataset:\n")
    print(final_df.tail())
