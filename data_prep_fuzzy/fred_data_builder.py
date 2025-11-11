import os
import pandas as pd
import numpy as np
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("FRED_API_KEY"))
fred = Fred(os.getenv("FRED_API_KEY"))

# Define all Fred Indidicators and metadata used in static model
FRED_SERIES = {
    # Rates, Spreads, Inflation
    "DGS10": {"desc": "10Y Treasury Yield", "freq": "D", "window": 120}, # 10 years
    "DGS2": {"desc": "2Y Treasury Yield", "freq": "D", "window": 60}, # 5 years
    "BAMLH0A0HYM2": {"desc": "High-Yield Credit Spread", "freq": "D", "window": 60}, # 5 years
    "FEDFUNDS": {"desc": "Effective Fed Funds Rate", "freq": "M", "window": 60}, # 5 years
    "CPIAUCSL": {"desc": "CPI (All Urban Consumers)", "freq": "M", "window": 36}, # 3 years
    # Growth and Production
    "A191RL1Q225SBEA": {"desc": "Real GDP Growth Rate", "freq": "Q", "window": 60}, # 5 years
    "INDPRO": {"desc": "Industrial Production Index", "freq": "M", "window": 36}, # 3 years
    "PAYEMS": {"desc": "Nonfarm Payrolls", "freq": "M", "window": 36}, # 3 years
    "CP": {"desc": "Corporate Profits", "freq": "Q", "window": 60}, # 5 years
    # Sentiment, Volatility, Liquidity
    "UMCSENT": {"desc": "Consumer Sentiment", "freq": "M", "window": 36}, # 3 years
    "VXOCLS": {"desc": "VIX (CBOE Volatility Index)", "freq": "D", "window": 36}, # 3 years
    "M2SL": {"desc": "Money Supply M2", "freq": "M", "window": 36}, # 3 years
    "USEPUINDXD": {"desc": "Economic Policy Uncertainty Index", "freq": "M", "window": 36}, # 3 years
    # Capex proxy
    "PNFIC1": {"desc": "Private Nonresidential Fixed Investment", "freq": "Q", "window": 60}, # 5 years
    "GDP": {"desc": "Nominal GDP", "freq": "Q", "window": 60}, # 5 years
}

START_DATE = '2000-01-01'
END_DATE = '2023-12-31'
DATA_DIR = "fred"
os.makedirs(DATA_DIR, exist_ok=True)

# Pull down and resample each series from above into a monthly frequency
def fetch_and_resample(code, meta):
    print(f"Fetching {code}: {meta['desc']}")
    series = fred.get_series(code, observation_start=START_DATE, observation_end=END_DATE)
    if series is None:
        print(f"Warning: {code} is 'None.'")
    elif series.empty:
        print(f"Warning: {code} is 'empty.'")
    
    # resample to month end, some series are daily, sparse, or quarterly and need to be upsampled to monthly
    if meta["freq"] in ["D", "B"]:
        s = series.resample("ME").last()
    elif meta["freq"] == "Q":
        s = series.resample("QE").last().resample("ME").ffill() #upsampling quartly data monthly
    else:
        s = series.resample("ME").last()
    
    s.name = code
    return s

# Build the raw dataset by looping through the fred series and producing a csv
def build_raw_dataset():
    series_list = []
    for code, meta in FRED_SERIES.items(): 
        s = fetch_and_resample(code, meta)
        if s is not None:
            series_list.append(s)
    
    df = pd.concat(series_list, axis = 1)
    df = df.dropna(how="all")
    df.to_csv(os.path.join(DATA_DIR, "raw_fred_data.csv"))
    print(f"\nSaved raw monthly FRED dataset to {DATA_DIR}/raw_fred_data.csv ({df.shape[0]} rows)\n... Check for NAN values")
    return df

# compute rolling z-score for each one with defined windows above. 
# to edit look back windows from raw data for each element of the series
# simply change the lookback value (units of months)
def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def zscore_all(df: pd.DataFrame):
    df_z = pd.DataFrame(index=df.index)
    for code, meta in FRED_SERIES.items():
        if code not in df.columns:
            continue
        w = meta.get("window", 36) #lookback window, defualted to 36 month if not defined
        df_z[f"{code}_z"] = rolling_zscore(df[code], w)
    df_z = df_z.dropna(how="all")
    df_z.to_csv(os.path.join(DATA_DIR, "zscored_raw.csv"))
    print(f"Saved z-scored dataset to {DATA_DIR}/zcored_raw.csv ({df_z.shape[1]} columns)")
    return df_z

# Main
if __name__ == "__main__":
    print("Building unified FRED dataset...\n")
    raw_df = build_raw_dataset()
    print(type(raw_df))
    if raw_df is None:
        print("raw_df is None — check FRED fetches or return statement.")
    else:
        print(raw_df.shape)
        print(raw_df.head())
    z_df = zscore_all(raw_df)
    print("\nPreview of z-scored data:")
    print(z_df.tail(5))
