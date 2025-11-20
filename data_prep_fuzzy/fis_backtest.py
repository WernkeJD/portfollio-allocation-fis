# fis_backtest.py

import yfinance as yf
import pandas as pd
import numpy as np
import json

from factor_panels import load_zscored_fred, build_fis_factor_panels
from fuzzy_allocator import FuzzyAllocator



# unchanged backtest logic
class Portfolio:
    def __init__(self, starting_value: float = 1000000):
        self.starting_value = starting_value
        self.start_date = "2007-04-30"
        self.end_date = "2021-07-31"

    def align_to_common_index(self, dfs):
        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)
        return [df.loc[common_index].copy() for df in dfs], common_index

    def generate_sp5mv_df(self, filepath): # create the sp5mv df using the json
        with open(filepath, "r") as f:
            data = json.load(f)
        prices = {
            item["attributes"]["as_of_date"]: item["attributes"]["close"]
            for item in data["data"]
        }
        df = pd.DataFrame(
            {"price": prices.values()},
            index=pd.to_datetime(list(prices.keys()))
        )
        return df

    def get_price_data(self, tickers, csv_output=False):
        """
        Download ETF prices with yfinance and append SP5MV from json.
        """
        sp5mv = self.generate_sp5mv_df("sp5mv.json").resample("ME").last() #uses json

        data = yf.download(tickers, start=self.start_date, end=self.end_date)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        data.index = pd.to_datetime(data.index)
        data = data.resample("ME").last()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        aligned, common_index = self.align_to_common_index([data, sp5mv])
        prices = aligned[0].copy()
        prices["SP5MV"] = aligned[1]["price"]


        if csv_output:
            prices.to_csv("aligned_price_data.csv") # returns a functionally identical csv to the OG portfolio code
                                                    # differences: Monthly sampling, includes vflq (unused), starts a little later than OG but works fine

        return [prices], common_index

    def generate_allocation(self, prices_row, weights_row):
        alloc = {}
        shares = {}
        for t in weights_row.index:
            dollars = float(weights_row[t] * self.starting_value)
            price = float(prices_row[t])
            alloc[t] = dollars
            shares[t] = dollars / price if price > 0 else 0.0
        return alloc, shares

    def backtest(self, prices, weights_df, rebalance_freq="ME"):
        prices = prices.resample("ME").last()
        aligned, common_index = self.align_to_common_index([prices, weights_df])
        prices, weights_df = aligned
        prices = prices.sort_index()
        weights_df = weights_df.sort_index()

        tickers = [t for t in prices.columns if t in weights_df.columns]
        if "^SPX" not in prices.columns:
            raise ValueError("Expected '^SPX' in price data.")

        first_date = common_index[0]
        snp_price0 = float(prices.loc[first_date]["^SPX"])
        snp_shares = self.starting_value / snp_price0

        shares_owned = {t: 0.0 for t in tickers}
        portfolio_value = self.starting_value

        results = []

        for i, date in enumerate(common_index):
            price_row = prices.loc[date]
            weight_row = weights_df.loc[date]

            if i > 0:
                portfolio_value = sum(
                    shares_owned[t] * float(price_row[t]) for t in tickers
                )

            new_alloc = {}
            new_shares = {}
            for t in tickers:
                dollars = float(weight_row[t]) * portfolio_value
                new_alloc[t] = dollars
                new_shares[t] = (
                    dollars / float(price_row[t]) if price_row[t] > 0 else 0.0
                )

            shares_owned = new_shares

            snp_price = float(price_row["^SPX"])
            snp_value = snp_shares * snp_price
            snp_return = snp_value / self.starting_value - 1.0

            results.append({
                "date": date,
                "port_allocation": new_alloc,
                "shares_owned": new_shares,
                "port_value": portfolio_value,
                "snp_price": snp_price,
                "snp_value": snp_value,
                "snp_return": snp_return,
            })

        return pd.DataFrame(results).set_index("date")



# main: zscores -> FIS -> tilts -> backtest
if __name__ == "__main__":
    port = Portfolio()

    # -----------------------------
    # Toggles
    # -----------------------------
    USE_FUZZY   = True
    PLOT_STATIC = True
    PRINT_DIFF  = True

    # -----------------------------
    # 1. Load price data for investable ETFs
    # -----------------------------
    # IMPORTANT: SPX is investable. ^SPX is the benchmark only.
    investable = ["VYM", "IVW", "PDP", "^SPX"]
    price_frames, _ = port.get_price_data(investable, csv_output=False)

    prices_df = price_frames[0].resample("ME").last()
    common_index = prices_df.index

    # -----------------------------
    # 2. Build factors and align
    # -----------------------------
    fred_z = load_zscored_fred("fred/zscored_clipped.csv")
    factor_panels = build_fis_factor_panels(fred_z)

    for k in factor_panels.keys():
        df = factor_panels[k].resample("ME").last()
        factor_panels[k] = df.loc[common_index].ffill().bfill()

    # -----------------------------
    # 3. Compute weights
    # -----------------------------
    # FULL INVESTABLE UNIVERSE:
    etfs_traded = ["VYM", "IVW", "PDP", "^SPX", "SP5MV"]

    if USE_FUZZY:
        print("\nUsing fuzzy allocator...\n")
        fuzzy = FuzzyAllocator(factor_panels)
        tilts_df = fuzzy.build_tilts().reindex(common_index).ffill().bfill()
        weights_df = tilts_df.div(tilts_df.sum(axis=1), axis=0)

    else:
        print("\nUsing static equal-weight 20%...\n")
        weights_df = pd.DataFrame(
            1.0 / len(etfs_traded),
            index=common_index,
            columns=etfs_traded
        )

    # -----------------------------
    # 4. Backtest
    # -----------------------------
    portfolio_returns = port.backtest(prices=prices_df, weights_df=weights_df)
    print("\nPortfolio returns:\n")
    print(portfolio_returns)

    # -----------------------------
    # 5. Allocation Plot
    # -----------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    weights_df.plot.area(colormap="tab20", linewidth=0)
    plt.title("Portfolio Allocation Breakdown Over Time")
    plt.ylabel("Portfolio Weight")
    plt.xlabel("Date")
    plt.legend(loc="upper left", ncol=3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 6. Performance Curves
    # -----------------------------
    fuzzy_curve = portfolio_returns["port_value"]
    spx_curve   = portfolio_returns["snp_value"]  # benchmark (^SPX)

    plt.figure(figsize=(14, 6))
    plt.plot(fuzzy_curve, label="Fuzzy Portfolio", linewidth=2)
    plt.plot(spx_curve, label="S&P 500 Benchmark (^SPX)", linewidth=2)

    # Add static curve if comparing
    if PLOT_STATIC and USE_FUZZY:
        static_w = pd.DataFrame(
            1.0 / len(etfs_traded),
            index=common_index,
            columns=etfs_traded
        )
        static_returns = port.backtest(prices=prices_df, weights_df=static_w)
        plt.plot(static_returns["port_value"], label="Static EW Portfolio", linewidth=2)

    plt.title("Portfolio Value vs Benchmark")
    plt.ylabel("Value ($)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 7. Print % difference vs SPX
    # -----------------------------
    if PRINT_DIFF:
        fuzzy_final  = fuzzy_curve.iloc[-1]
        spx_final    = spx_curve.iloc[-1]

        diff_percent = (fuzzy_final - spx_final) / spx_final * 100
        print(f"\nFuzzy Portfolio vs SPX Final Return Difference: {diff_percent:.2f}%")

        if USE_FUZZY:
            static_final = static_returns["port_value"].iloc[-1]
            diff_percent_static = (static_final - spx_final) / spx_final * 100
            print(f"Static EW Portfolio vs SPX Final Return Difference: {diff_percent_static:.2f}%")
