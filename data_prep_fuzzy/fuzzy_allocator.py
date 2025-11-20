# fuzzy_allocator.py

from __future__ import annotations
import pandas as pd

from fis_arch import (
    build_vym_fis,
    build_ivw_fis,
    build_pdp_fis,
    build_spx_fis,
    build_sp5mv_fis,
)


class FuzzyAllocator:
    """
    Runs Mamdani FIS blocks over z-scored factor panels and
    returns tilt time series for each ETF.

    factor_panels must be:
      {
        "VYM":  df with cols [y10y, cpi, gdpg, hyspd],
        "IVW":  df with cols [spread, realff, eps, capex],
        "PDP":  df with cols [vix, m2, spd, inp, cnsm],
        "SPX":  df with cols [esi, vix, cnsm, m2, nfp],
        "SP5MV": df with cols [vix, rgdp, cspd, cnsm, unmpr],
      }

    All DataFrames must share a DateTimeIndex (monthly).
    """

    def __init__(self, factor_panels: dict[str, pd.DataFrame]):
        self.factor_panels = factor_panels

    @staticmethod
    def _run_fis_over_df(build_fis_fn, df: pd.DataFrame, input_names: list[str]) -> pd.Series:
        """
        Reuse a single ControlSystemSimulation, feeding inputs row-by-row.
        """
        tilts = []

        sim = build_fis_fn()
        for dt, row in df.iterrows():
            sim.reset()
            for name in input_names:
                sim.input[name] = float(row[name])
            sim.compute()
            tilts.append((dt, float(sim.output["tilt"])))

        return pd.Series(dict(tilts))

    def build_tilts(self) -> pd.DataFrame:
        """
        Run all available FIS and return a tilt DataFrame with columns:
          VYM, IVW, PDP, ^SPX, SP5MV
        """
        tilt_series: dict[str, pd.Series] = {}

        # VYM
        if "VYM" in self.factor_panels:
            df = self.factor_panels["VYM"]
            s = self._run_fis_over_df(build_vym_fis, df, ["y10y", "cpi", "gdpg", "hyspd"])
            tilt_series["VYM"] = s

        # IVW
        if "IVW" in self.factor_panels:
            df = self.factor_panels["IVW"]
            s = self._run_fis_over_df(build_ivw_fis, df, ["spread", "realff", "eps", "capex"])
            tilt_series["IVW"] = s

        # PDP
        if "PDP" in self.factor_panels:
            df = self.factor_panels["PDP"]
            s = self._run_fis_over_df(build_pdp_fis, df, ["vix", "m2", "spd", "inp", "cnsm"])
            tilt_series["PDP"] = s

        # SPX (store as ^SPX to match price data)
        if "SPX" in self.factor_panels:
            df = self.factor_panels["SPX"]
            s = self._run_fis_over_df(build_spx_fis, df, ["esi", "vix", "cnsm", "m2", "nfp"])
            tilt_series["^SPX"] = s

        # SP5MV
        if "SP5MV" in self.factor_panels:
            df = self.factor_panels["SP5MV"]
            s = self._run_fis_over_df(build_sp5mv_fis, df, ["vix", "rgdp", "cspd", "cnsm", "unmpr"])
            tilt_series["SP5MV"] = s

        if not tilt_series:
            raise ValueError("No factor panels provided to FuzzyAllocator.")

        return pd.DataFrame(tilt_series).sort_index()
