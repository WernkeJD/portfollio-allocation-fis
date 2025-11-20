from dotenv import load_dotenv 
from fredapi import Fred
import pandas as pd
import numpy as np
import os
import yfinance as yf
import json
import requests

# from portfolio import Portfolio

load_dotenv()
fred_key = os.getenv("FRED-KEY")
fred = Fred(api_key=fred_key)

"""
VYM (Dividend Yeild ETF):
    Factors:
        10y treasury yeild: Negative Relationship
        CPI: positive up to 3% negative beyond
        GDP Growth: Negative
        High-Yield Credit Spread: Positive Relationship


IVW (Growth ETF):
    Factors:
        10y-2y Treasury Spread: Positive Relationship
        Real Fed Funds: Negative Relationship
        Corporate Profits (EPS Growth Proxy): Positive Relationship
        Capex_ratio: Positve Relationship  

PDP (Momentum ETF): Switched from MTUM cause the price history did not go back beyond 2013 however other ETF's could be used.
    Factors:
        VIX: 
        Trading volume: 
        Cross-sectional volatility: 
        Momentum factor return:

SP5MV(Volitility Index):
    Factors:
        Volatility (VIX): positive correlation
        Real GDP growth: negative correlation
        Consumer sentiment: negative correlation
        Credit spreads are wide: positive correlation (higher spreads higher stress)
        Unemployment: positive correlation


SPX (Beta ETF):
    Factors:
        Economic Policy Unscertanty: negative correlation
        VIX: negative correlation more volitility eats into beta
        Consumer Sentiment: Positive correlation more market confidence
        M2 Growth: Positive correlation more money in supply is more for consumers to spend presumably
        NFP changes:Positive correlation more job growth is more consumer spending good for beta

VFLQ (Liquidity ETF):
    factors:
        Effective Federal Funds Rate: negative correlation, lower funds rate means lower cost to borrow increases liquidity 
        Corporate Bond Spreads: negative correlation, widening spreads mean that investors are demanding more for risk 
        Retail trading volume: Positive correlation, more market participation benefits liquid stocks 
        Money Supply: postive correlation: more money means more free capital to trade
        Non farm payroll: positive correlation: Positive correlation, same reason as money supply
"""

class FactorAnalysis:
    def __init__(self):
        self.vym_desirability = None
        self.ivw_desirability = None
        self.mtum_desirability = None
        self.spx_desirability = None
        self.vflq_desirability = None
        self.start_date = '2000-01-01'
        self.end_date = '2023-12-31'

    #helpers
    def normalize(self, series: pd.Series):
        "Takes in a pandas series, historical facotr data in this case, and normalizes each value to a decimal between 0 and 1"
        return series.rank(pct=True)

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

    def output_data(self, data: pd.DataFrame, name: str):
        try:
            data.to_csv(name, index=True)
            return "data added"
        except Exception as e:
            return f"an error occured: {e}"


    #Calculations
    """
    each calculate function uses a set of factors with some significant corelation to the
    macro trend each etf tracks. they take in no values and return a dataframe that contains the normalized
    macro factor information, a score for the index, and a reccomended allocation. All data is resampled monthly,
    as such, each reccomendation allocation and score is also a reccomendation for the month.

    For information on the factors used for each etf's calculation, see the top of this calass.

    """
    def calculate_vym(self):

        data = {
            "10Y": fred.get_series("DGS10", observation_start = self.start_date, observation_end=self.end_date),                      # 10Y Treasury Yield
            "CPI": fred.get_series("CPIAUCSL", observation_start = self.start_date, observation_end=self.end_date),                   # CPI
            "GDP": fred.get_series("A191RL1Q225SBEA", observation_start = self.start_date, observation_end=self.end_date),            # Real GDP Growth %
            "Spread": fred.get_series("BAMLH0A0HYM2", observation_start = self.start_date, observation_end=self.end_date),            # High-Yield Credit Spread
        }

        df = pd.DataFrame(data).resample("ME").last().dropna()

        df_norm = df.apply(self.normalize)

        treasury = 1 - df_norm["10Y"]
        cpi = 1 - abs(df_norm["CPI"] - 0.5) * 2
        gdp = 1 - df_norm["GDP"]
        spread = df_norm["Spread"]

        df["VYM_score"] = (
            0.35 * treasury +
            0.15 * cpi +
            0.3  * gdp +
            0.2 * spread
        )

        df["VYM_allocation"] = 0.05 + 0.25 * df["VYM_score"]


        return df
    
    def calculate_ivw(self):
        dgs10 = fred.get_series("DGS10", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()
        dgs2  = fred.get_series("DGS2", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()
        ff    = fred.get_series("FEDFUNDS", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()
        cpi   = fred.get_series("CPIAUCSL", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()
        infl_yoy = cpi.pct_change(12) * 100

        # --- Quarterly -> Monthly (ffill) proxies ---
        # EPS proxy from BEA corporate profits (YoY)
        cp = fred.get_series("CP", observation_start = self.start_date, observation_end=self.end_date).resample("QE-DEC").last()
        eps_yoy_q = cp.pct_change(4) * 100
        eps_yoy_m = eps_yoy_q.resample("ME").ffill()

        capex = fred.get_series("PNFIC1", observation_start = self.start_date, observation_end=self.end_date).resample("QE-DEC").last()
        gdp   = fred.get_series("GDP", observation_start = self.start_date, observation_end=self.end_date).resample("QE-DEC").last()
        capex_ratio_q = (capex / gdp) * 100
        capex_ratio_m = capex_ratio_q.resample("ME").ffill()

        df = pd.concat({
            "yc_10y2y": dgs10 - dgs2,
            "real_ff":  ff - infl_yoy,
            "eps_yoy":  eps_yoy_m,
            "capex_gdp_pct": capex_ratio_m
        }, axis=1).dropna()

        df_norm = df.apply(self.normalize)

        spread_score  = df_norm["yc_10y2y"]       # steeper curve = better for growth
        realrate_scr  = 1 - df_norm["real_ff"]    # tighter real rate = worse (invert)
        eps_score     = df_norm["eps_yoy"]        # faster earnings proxy growth = better
        capex_score   = df_norm["capex_gdp_pct"]  # higher capex intensity = better

        df["IVW_score"] = (
            0.30 * spread_score +
            0.25 * realrate_scr +
            0.25 * eps_score +
            0.20 * capex_score
        ).clip(0, 1)

        df["IVW_allocation"] = 0.05 + 0.25 * df["IVW_score"]
        return df

    def calculate_pdp(self):
        # --- Fetch monthly data ---
        vix   = fred.get_series("VXOCLS", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()        # volatility proxy
        m2    = fred.get_series("M2SL", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()          # liquidity supply
        spread = fred.get_series("BAMLH0A0HYM2", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last() # credit spread
        indpro = fred.get_series("INDPRO", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()       # production volume
        umc   = fred.get_series("UMCSENT", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()       # sentiment

        # --- Transformations ---
        indpro_yoy = indpro.pct_change(12) * 100
        m2_yoy     = m2.pct_change(12) * 100

        df = pd.concat({
            "vol": vix,
            "liq_yoy": m2_yoy,
            "spread": spread,
            "indpro_yoy": indpro_yoy,
            "sent": umc
        }, axis=1).dropna()

        df_norm = df.apply(self.normalize)

        vol_score    = 1 - df_norm["vol"]
        liq_score    = df_norm["liq_yoy"]
        spread_score = 1 - df_norm["spread"]
        indpro_score = df_norm["indpro_yoy"]
        sent_score   = df_norm["sent"]

        df["PDP_score"] = (
            0.25 * vol_score +
            0.25 * liq_score +
            0.20 * spread_score +
            0.15 * indpro_score +
            0.15 * sent_score
        ).clip(0, 1)

        df["PDP_allocation"] = 0.05 + 0.25 * df["PDP_score"]
        return df
    
    def calculate_spx(self):
        esi   = fred.get_series("USEPUINDXD", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()       # economic surprise metric (for now)
        vix    = fred.get_series("VXOCLS", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()          # voitiity index
        con_sent = fred.get_series("UMCSENT", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()       # consumer sentiment metric
        m2 = fred.get_series("M2SL", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()                # measure of us money suppy
        nfp   = fred.get_series("PAYEMS", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()           # non farm payroll


        df = pd.concat({
            "esi": esi,
            "vix": vix,
            "con_sent": con_sent,
            "m2": m2,
            "nfp": nfp
        }, axis=1).dropna()

        df_norm = df.apply(self.normalize)

        esi_score    = 1 - df_norm["esi"]
        vix_score    = 1 - df_norm["vix"]
        con_sent_score = df_norm["con_sent"]
        m2_score = df_norm["m2"]
        nfp_score   = df_norm["nfp"]

        df["SPX_score"] = (
            0.25 * esi_score +
            0.25 * vix_score +
            0.20 * con_sent_score +
            0.15 * m2_score +
            0.15 * nfp_score
        ).clip(0, 1)

        df["SPX_allocation"] = 0.05 + 0.25 * df["SPX_score"]
        return df
    
    def calculate_VFLQ(self):
        # --- Fetch monthly data ---
        effr   = fred.get_series("DFF", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()             # effective fed funds rate
        corp_spread    = fred.get_series("AAA10Y", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()  # corporate bond spread
        retail_trading_volume = fred.get_series("USASLRTTO01GYSAM", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()             # snp 500 trading volume
        m2 = fred.get_series("M2SL", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()                # measure of us money suppy
        nfp   = fred.get_series("PAYEMS", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()           # non farm payroll


        # --- Combine into DataFrame ---
        df = pd.concat({
            "effr": effr,
            "corp_spread": corp_spread,
            "retail_trading_volume": retail_trading_volume,
            "m2": m2,
            "nfp": nfp
        }, axis=1).dropna()

        df_norm = df.apply(self.normalize)

        effr_score    = 1- df_norm["effr"]
        corp_spread_score    = 1 - df_norm["corp_spread"]
        retail_trading_volume = df_norm["retail_trading_volume"]
        m2_score = df_norm["m2"]
        nfp_score   = df_norm["nfp"]

        df["VFLQ_score"] = (
            0.25 * effr_score +
            0.25 * corp_spread_score +
            0.20 * retail_trading_volume +
            0.15 * m2_score +
            0.15 * nfp_score
        ).clip(0, 1)

        df["VFLQ_allocation"] = 0.05 + 0.25 * df["VFLQ_score"]
        return df

    def calculate_sp5mv(self):

        vix = fred.get_series("VIXCLS", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()
        real_gdp =  fred.get_series("GDPC1", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()
        credit_spreads =  fred.get_series("T10Y2Y", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()
        consumer_sentiment =  fred.get_series("UMCSENT", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()
        unemployment =  fred.get_series("UNRATE", observation_start = self.start_date, observation_end=self.end_date).resample("ME").last()

        df = pd.concat({
            "vix": vix,
            "rgdp": real_gdp,
            "credit_spreads": credit_spreads,
            "consumer_sentiment": consumer_sentiment,
            "unemployment": unemployment
        }, axis=1).dropna()

        df_norm = df.apply(self.normalize)

        vix_score    = df_norm["vix"]
        rgdp_score    = 1 - df_norm["rgdp"]
        credit_spread_score = df_norm["credit_spreads"]
        con_sent_score = 1- df_norm["consumer_sentiment"]
        unemployment_score   = df_norm["unemployment"]

        df["SP5MV_score"] = (
            0.25 * vix_score +
            0.10 * credit_spread_score +
            0.10 * con_sent_score +
            0.25 * rgdp_score +
            0.30 * unemployment_score
        ).clip(0, 1)

        df["SP5MV_allocation"] = 0.05 + 0.25 * df["SP5MV_score"]

        return df
    

    #Final allocation
    def build_portfolio_weights(
        self,
        use_vym=True,
        use_ivw=True,
        use_pdp=True,
        use_spx=True,
        use_vflq=True,
        use_sp5mv = True,
        floor: float = 0.0,
        cash_weight: float = 0.0,
    ) -> pd.DataFrame:
        """
        Build a joint portfolio weight matrix from ETF regime scores.
        Returns a DataFrame indexed by date, columns = tickers,
        where each row sums to (1 - cash_weight).

        floor: per-ETF base weight added before normalization (0–something).
        cash_weight: fixed fraction reserved for cash (0–1).
        """

        dfs = []
        cols = []

        if use_vym:
            vym = self.calculate_vym()
            dfs.append(vym[["VYM_score"]])
            cols.append("VYM")
        if use_ivw:
            ivw = self.calculate_ivw()
            dfs.append(ivw[["IVW_score"]])
            cols.append("IVW")
        if use_pdp:
            dwa = self.calculate_pdp()
            dfs.append(dwa[["PDP_score"]])
            cols.append("PDP")
        if use_spx:
            spx = self.calculate_spx()
            dfs.append(spx[["SPX_score"]])
            cols.append("^SPX")
        if use_vflq:
            vflq = self.calculate_VFLQ()
            dfs.append(vflq[["VFLQ_score"]])
            cols.append("VFLQ")
        if use_sp5mv:
            sp5mv = self.calculate_sp5mv()
            dfs.append(sp5mv[["SP5MV_score"]])
            cols.append("SP5MV")

        if not dfs:
            raise ValueError("No ETFs selected for portfolio weights.")

        # 1) Align all score DataFrames on common dates
        aligned_dfs, common_index = self.align_to_common_index(dfs)

        # 2) Combine into one score matrix
        scores = pd.concat(
            [df.rename(columns={df.columns[0]: col}) for df, col in zip(aligned_dfs, cols)],
            axis=1
        )

        # 3) Turn scores into raw signals
        signals = scores.clip(lower=0.0, upper=1.0)
        if floor > 0.0:
            raw = floor + signals
        else:
            raw = signals.copy()

        # 4) normalize weights so all rows sum to 1
        row_sums = raw.sum(axis=1).replace(0.0, np.nan)
        weights = raw.div(row_sums, axis=0)
        if cash_weight > 0.0:
            weights = weights * (1.0 - cash_weight)
            weights["CASH"] = cash_weight

        weights = weights.dropna(how="any")

        return weights

    

if __name__ == "__main__":
    run = True
    analysis = FactorAnalysis()
    port = Portfolio()

    if run:
        # Step 1: Build the joint portfolio weights 
        print("\nBuilding joint portfolio weights for VYM, IVW, DWA, VFLQ, and SPX...")
        weights_df = analysis.build_portfolio_weights(
            use_vym=True,
            use_ivw=True,
            use_pdp=True,
            use_spx=True,
            use_vflq=True,
            use_sp5mv = True,
            floor=0.05,       
            cash_weight=0.00  
        )

        # Print summary
        print("\nSample of combined portfolio weights:")
        print(weights_df.head())

        print("\nWeight sums per row (should be 1.00):")
        print(weights_df.sum(axis=1).head())

        print("\nDate range covered:", weights_df.index.min(), "→", weights_df.index.max())
        print("Columns included:", list(weights_df.columns))

        # Optional Step 3: Export for backtesting
        weights_df.to_csv("portfolio_weights.csv")
        print("\nSaved portfolio weights to 'portfolio_weights.csv'.")


        