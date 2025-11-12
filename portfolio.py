import yfinance as yf
import pandas as pd
import numpy as np
import json

class Portfolio:
    def __init__(self, starting_value: float=1000000):
        self.starting_value = starting_value
        self.total_value = starting_value
        self.positions = None
        self.value_history = None
        self.start_date = "2007-04-30"
        self.end_date = "2021-07-31"

    def align_to_common_index(self, dfs: list[pd.DataFrame]):
        """
        Take a list of DataFrames and return:
          (aligned_dfs, common_index)
        where aligned_dfs are restricted to the intersection of all indices.
        """

        if not dfs:
            return [], None


        common_index = dfs[0].index
        for df in dfs[1:]:
            common_index = common_index.intersection(df.index)

        aligned = [df.loc[common_index].copy() for df in dfs]
        return aligned, common_index

    def generate_sp5mv_df(self, filepath: str):
        with open(file=filepath, mode='r') as file:
            data = json.load(file)

        prices = {}

        for item in data["data"]:
            price = item["attributes"]["close"]
            date = item["attributes"]["as_of_date"]

            prices[date] = price

        
        df = pd.DataFrame(data=prices.values(), index=prices.keys(), columns=["price"])
        return df

    def get_price_data(self, tickers: list[str], csv_output = False) -> pd.DataFrame:        
        """
        Download adjusted close prices for the given tickers between start and end.
        """

        sp5mv = self.generate_sp5mv_df(filepath="sp5mv.json")
        sp5mv.index = pd.to_datetime(sp5mv.index)

        data = yf.download(tickers, start=self.start_date, end=self.end_date)["Close"]

        #converts to df is only one ticker is provided since default return from pd is Series
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        aligned_df, common_index = self.align_to_common_index(dfs=[data, sp5mv])

        aligned_df[0]["sp5mv"] = aligned_df[1]["price"]

        if csv_output:
            aligned_df[0].to_csv("aligned_price_data.csv")

        return aligned_df, common_index

    def generate_starting_portfolio(self, prices_row: pd.Series, weights_row: pd.Series):
        """
        Given a first date, a row of prices, and a row of target weights,
        compute initial share positions.
        """

        weights = weights_row.fillna(0.0)
        if weights.sum() > 0:
            weights = weights / weights.sum()

        else:
            self.positions = {t: 0.0 for t in prices_row.index}
            self.total_value = self.starting_value
            return
        
        dollar_alloc = self.starting_value * weights
        shares = dollar_alloc / prices_row
        self.positions = shares.to_dict()
        self.total_value = self.starting_value


    def backtest(self, prices: pd.DataFrame, weights_df: pd.DataFrame, rebalance_freq: str = "ME") -> pd.Series:
        """
        Backtest the portfolio over the given price history, rebalancing according to weights_df.

        prices: DataFrame of Adj Close prices (index: dates, columns: tickers)
        weights_df: DataFrame of target weights (index: dates, columns: tickers)
                    The index should be a subset of prices.index (e.g., month end).
        rebalance_freq: not strictly used yet (weights_df already encodes rebalancing dates),
                        but kept for future flexibility.
        """
        prices = prices.sort_index()
        weights_df = weights_df.sort_index()

        # Ensure columns match (same tickers)
        tickers = prices.columns
        weights_df = weights_df.reindex(columns=tickers).fillna(0.0)

        # Use only dates where we have price data
        weights_df = weights_df.loc[weights_df.index.intersection(prices.index)]

        if weights_df.empty:
            raise ValueError("weights_df and prices have no overlapping dates.")

        # Our rebalancing dates (when we change weights)
        rebalance_dates = list(weights_df.index)

        # Initialize
        first_reb_date = rebalance_dates[0]
        # Align first rebalance date to the closest price date >= first_reb_date
        first_price_idx = prices.index.searchsorted(first_reb_date)
        if first_price_idx == len(prices.index):
            raise ValueError("First rebalance date is after the last price date.")
        current_date = prices.index[first_price_idx]

        # Initialize positions at first rebalance
        self.generate_starting_portfolio(
            prices_row=prices.loc[current_date],
            weights_row=weights_df.loc[first_reb_date]
        )

        portfolio_values = pd.Series(index=prices.index, dtype=float)
        current_value = self.starting_value

        # Pointer to next rebalance date
        reb_idx = 1 if len(rebalance_dates) > 1 else None
        next_reb_date = rebalance_dates[reb_idx] if reb_idx is not None else None

        for date in prices.index:
            # If we reached or passed the next rebalance date, rebalance
            if next_reb_date is not None and date >= next_reb_date:
                # Compute portfolio value at this date before rebalancing
                prices_today = prices.loc[date]
                current_value = sum(self.positions[t] * prices_today[t] for t in tickers)

                # Get new target weights
                w = weights_df.loc[next_reb_date].fillna(0.0)
                if w.sum() > 0:
                    w = w / w.sum()

                dollar_alloc = current_value * w
                shares = dollar_alloc / prices_today
                self.positions = shares.to_dict()

                # Advance to next rebalance date if any
                reb_idx = reb_idx + 1 if reb_idx is not None else None
                if reb_idx is not None and reb_idx < len(rebalance_dates):
                    next_reb_date = rebalance_dates[reb_idx]
                else:
                    next_reb_date = None  # no more rebalances

            # Mark-to-market portfolio value for this date
            prices_today = prices.loc[date]
            portfolio_values.loc[date] = sum(self.positions[t] * prices_today[t] for t in tickers)
            current_value = portfolio_values.loc[date]

        self.value_history = portfolio_values
        self.total_value = current_value
        return portfolio_values

    @staticmethod
    def compute_returns(portfolio_values: pd.Series) -> pd.Series:
        """
        Compute simple returns from a portfolio value series.
        """
        returns = portfolio_values.pct_change().dropna()
        return returns

if __name__ == "__main__":

    port = Portfolio()
    json_data = port.generate_sp5mv_df("sp5mv.json")
    print(json_data)
    tickers, common_index = port.get_price_data(["VYM", "IVW", "PDP", "^SPX", "VFLQ", "USMV"], csv_output=True)

    print("tickers \n", tickers, "\n")
    print(type(tickers))
    print("index \n", common_index, "\n")
    print(type(common_index))

    # print("portfolio price data example\n")
    # print("df head: ")
    # print(tickers.head(), "\n")
    # print("df tail: ")
    # print(tickers.tail())

    # test = port.generate_starting_portfolio({"^GSPC": .25, "AMD": .75})
    # print(test)

