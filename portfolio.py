import yfinance as yf
import pandas as pd
import numpy as np
import json

from factors import FactorAnalysis

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

        aligned_df[0]["SP5MV"] = aligned_df[1]["price"]

        if csv_output:
            aligned_df[0].to_csv("csvs/aligned_price_data.csv")

        return aligned_df, common_index
    
    def compute_portfolio_risk(
        self,
        backtest_df: pd.DataFrame,
        alpha: float = 0.95,
        window: int = 52,
    ) -> pd.DataFrame:
        """
        Computes dollar-based AND percentage-based portfolio risk metrics.

        Adds:
            port_return          (pct)
            drawdown_pct         (pct)
            drawdown_dollar      (absolute $)
            max_drawdown_pct     (pct)
            max_drawdown_dollar  (absolute $)
            var_pct              (pct VaR)
            es_pct               (pct ES)
            var_dollar           ($ VaR)
            es_dollar            ($ ES)
        """

        df = backtest_df.copy()
        df = df.sort_index()

        df["port_value"] = pd.to_numeric(df["port_value"], errors="coerce")

        # === Returns ===
        df["port_return"] = df["port_value"].pct_change()

        # === Drawdown (percentage) ===
        running_peak = df["port_value"].cummax()
        df["drawdown_pct"] = df["port_value"] / running_peak - 1.0
        df["max_drawdown_pct"] = df["drawdown_pct"].cummin()

        # === Drawdown (dollar) ===
        df["drawdown_dollar"] = df["port_value"] - running_peak
        df["max_drawdown_dollar"] = df["drawdown_dollar"].cummin()

        # === Rolling VaR / ES (percentage form first) ===
        roll = df["port_return"].rolling(window=window, min_periods=20)

        # VaR as a % loss
        df["var_pct"] = -roll.quantile(1.0 - alpha)

        # ES helper
        def _rolling_es(returns: np.ndarray, alpha_inner: float) -> float:
            s = pd.Series(returns).dropna()
            if len(s) == 0:
                return np.nan
            q = s.quantile(1.0 - alpha_inner)
            tail = s[s <= q]
            if len(tail) == 0:
                return np.nan
            return -tail.mean()

        df["es_pct"] = roll.apply(lambda x: _rolling_es(x, alpha), raw=True)

        # === Convert VaR/ES from pct → dollars ===
        df["var_dollar"] = df["var_pct"] * df["port_value"]
        df["es_dollar"] = df["es_pct"] * df["port_value"]

        return df

    def generate_allocation(self, prices_row: pd.Series, weights_row: pd.Series):
        """
        Given a first date, a row of prices, and a row of target weights,
        compute initial share positions.
        """

        starting_port = {}
        shares_owned = {}

        for key, value in prices_row.items():
            starting_port[key] = float(weights_row[key] * self.starting_value)
            shares_owned[key] = float(starting_port[key] / prices_row[key])
            

        return starting_port, shares_owned


    # def backtest(self, prices: pd.DataFrame, weights_df: pd.DataFrame, rebalance_freq: str = "ME") -> pd.Series:


    #     backtest_data_init = pd.DataFrame(index=prices.index, columns=["port_allocation", "shares_owned", "snp_price", "snp_returns"])
        
    #     aligned_frames, common_index = self.align_to_common_index([weights_df, backtest_data_init])

    #     backtest_data = aligned_frames[1]

    #     for item in prices.index:
    #         backtest_data.at[item,"snp_price"] = prices.loc[item]["^SPX"]
    #         backtest_data.at[item,"snp_returns"] = ((self.starting_value / prices.loc[item]["^SPX"])) # this is so wrong but it's just placeholder data for now.
    #         if item in weights_df.index:
    #             allocation, shares_owned = self.generate_allocation(prices_row = prices.loc[item], weights_row =weights_df.loc[item]) 
    #             backtest_data.at[item, "port_allocation"] = allocation
    #             backtest_data.at[item, "shares_owned"] = shares_owned



    #     return backtest_data
    def backtest(self, prices: pd.DataFrame, weights_df: pd.DataFrame, rebalance_freq: str = "ME") -> pd.DataFrame:
        """
        Backtest the factor-based portfolio.

        - Resamples prices to the rebalance frequency (default: month-end "ME").
        - Aligns prices and weights on their common date index.
        - Rebalances the portfolio at each rebalance date to target weights.
        - Tracks SPX as a benchmark.

        Returns a DataFrame indexed by date with:
        - port_allocation: dict[ticker -> dollars]
        - shares_owned:   dict[ticker -> shares]
        - port_value:     float (total portfolio value)
        - snp_price:      float (^SPX price)
        - snp_value:      float (value of SPX benchmark)
        - snp_return:     float (cumulative return of SPX vs starting_value)
        """

        # 1) Resample prices to rebalance frequency (monthly end)
        if rebalance_freq == "ME":
            prices = prices.resample("ME").last()
        else:
            # you can extend this later for other frequencies
            prices = prices.resample(rebalance_freq).last()

        # 2) Align prices and weights on common dates
        aligned_frames, common_index = self.align_to_common_index([prices, weights_df])
        prices_aligned, weights_aligned = aligned_frames

        # Ensure dates are sorted
        prices_aligned = prices_aligned.sort_index()
        weights_aligned = weights_aligned.sort_index()
        common_index = prices_aligned.index

        # 3) Prepare result DataFrame
        result_cols = [
            "port_allocation",
            "shares_owned",
            "port_value",
            "snp_price",
            "snp_value",
            "snp_return",
        ]
        backtest_data = pd.DataFrame(index=common_index, columns=result_cols, dtype=object)

        # 4) Determine which tickers are actually in both prices and weights
        tickers = [c for c in prices_aligned.columns if c in weights_aligned.columns]

        if "^SPX" not in prices_aligned.columns:
            raise ValueError("Expected '^SPX' column in prices for benchmark tracking.")

        # 5) Setup initial portfolio & SPX benchmark
        portfolio_value = self.starting_value
        first_date = common_index[0]
        first_price_row = prices_aligned.loc[first_date]

        snp_price0 = float(first_price_row["^SPX"])
        snp_shares = self.starting_value / snp_price0  # buy-and-hold benchmark

        # Initially, no holdings before first rebalance
        shares_owned = {t: 0.0 for t in tickers}

        # 6) Main backtest loop
        for i, date in enumerate(common_index):
            price_row = prices_aligned.loc[date]
            weight_row = weights_aligned.loc[date]

            # Mark-to-market the existing portfolio (after the first date)
            if i == 0:
                portfolio_value = float(self.starting_value)
            else:
                portfolio_value = float(
                    sum(shares_owned[t] * float(price_row[t]) for t in tickers)
                )

            # Rebalance to target weights for this date
            target_alloc = {}
            new_shares = {}

            for t in tickers:
                price = float(price_row[t])
                w = float(weight_row[t])

                dollars = w * portfolio_value
                target_alloc[t] = dollars
                new_shares[t] = dollars / price if price > 0 else 0.0

            shares_owned = new_shares

            # SPX benchmark tracking
            snp_price = float(price_row["^SPX"])
            snp_value = float(snp_shares * snp_price)
            snp_return = snp_value / self.starting_value - 1.0

            # Store results
            backtest_data.at[date, "port_allocation"] = target_alloc
            backtest_data.at[date, "shares_owned"] = new_shares
            backtest_data.at[date, "port_value"] = float(portfolio_value)
            backtest_data.at[date, "snp_price"] = snp_price
            backtest_data.at[date, "snp_value"] = snp_value
            backtest_data.at[date, "snp_return"] = snp_return

        return backtest_data

if __name__ == "__main__":

    port = Portfolio()
    analysis = FactorAnalysis()
    json_data = port.generate_sp5mv_df("sp5mv.json")
    print(json_data)
    tickers, common_index = port.get_price_data(["VYM", "IVW", "PDP", "^SPX"], csv_output=True)

    weights_df = analysis.build_portfolio_weights(
            use_vym=True,
            use_ivw=True,
            use_pdp=True,
            use_spx=True,
            use_vflq=False,
            use_sp5mv = True,
            floor=0.05,       
            cash_weight=0.00  
        )


    prices_df = tickers[0]
    print("prices: \n",prices_df)
    prices_row = prices_df.iloc[0]
    weights_row = weights_df.iloc[0]
    # print("prices row maybe? ", prices_row)
    # print("weights row maybe? ", weights_row)
    
    starting_port, shares =  port.generate_allocation(prices_row=prices_row, weights_row=weights_row)


    # print(starting_port)
    # print(shares)

    portfolio_returns = port.backtest(prices = prices_df, weights_df = weights_df)
    print("portfolio returns \n", portfolio_returns)
    analysis.output_data(portfolio_returns,"csvs/aligned_price_data.csv")

    risk_df = port.compute_portfolio_risk(
    portfolio_returns,
    alpha=0.95,  # 95% VaR/ES
    window=52    # e.g. 52 weeks or 52 periods
    )

    risk_df.to_csv("csvs/portfolio_risk_metrics.csv")



    # print("portfolio price data example\n")
    # print("df head: ")
    # print(tickers.head(), "\n")
    # print("df tail: ")
    # print(tickers.tail())

    # test = port.generate_starting_portfolio({"^GSPC": .25, "AMD": .75})
    # print(test)

