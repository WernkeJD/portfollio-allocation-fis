from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np

from fis_arch import z_to_universe



# Load cleaned, zscored, clipped macro dataset
def load_zscored_fred(path: str | Path = "fred/zscored_clipped.csv") -> pd.DataFrame:
    """
    Load the final z-scored FRED dataset produced by fred_data_builder.py.
    Must contain already derived YC_z, RFF_z, UNRATE_z, etc.
    """
    df = pd.read_csv(path)

    # detect date column
    for c in ("Date", "date", "Unnamed: 0"):
        if c in df.columns:
            date_col = c
            break
    else:
        raise ValueError("CSV must contain a date column.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    return df


def _clip_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all inputs are clipped to [-2.5, 2.5] for the FIS universe."""
    return df.apply(z_to_universe)


# Build FIS-ready factor panels
def build_fis_factor_panels(df_z: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build one DataFrame per ETF with columns matching the FIS input names.

    FIS Inputs:
      VYM   : y10y, cpi, gdpg, hyspd
      IVW   : spread, realff, eps, capex
      PDP   : vix, m2, spd, inp, cnsm
      SPX   : esi, vix, cnsm, m2, nfp
      SP5MV : vix, rgdp, cspd, cnsm, unmpr
    """
    panels: dict[str, pd.DataFrame] = {}

    # VYM
    vym = pd.DataFrame(index=df_z.index)
    vym["y10y"]  = df_z["DGS10_z"]
    vym["cpi"] = df_z["CPI_YoY_z"]
    vym["gdpg"]  = df_z["A191RL1Q225SBEA_z"]
    vym["hyspd"] = df_z["BAMLH0A0HYM2_z"]
    panels["VYM"] = _clip_universe(vym)

    # IVW
    ivw = pd.DataFrame(index=df_z.index)
    ivw["spread"] = df_z["YC_z"]
    ivw["realff"] = df_z["RFF_z"]
    ivw["eps"]    = df_z["CP_z"]
    ivw["capex"]  = df_z["PNFIC1_z"]
    panels["IVW"] = _clip_universe(ivw)

    # PDP
    pdp = pd.DataFrame(index=df_z.index)
    pdp["vix"]  = df_z["VIXCLS_z"]
    pdp["m2"]   = df_z["M2SL_z"]
    pdp["spd"]  = df_z["BAMLH0A0HYM2_z"]
    pdp["inp"]  = df_z["INDPRO_z"]
    pdp["cnsm"] = df_z["UMCSENT_z"]
    panels["PDP"] = _clip_universe(pdp)

    # SPX
    spx = pd.DataFrame(index=df_z.index)
    spx["esi"]  = df_z["USEPUINDXD_z"]
    spx["vix"]  = df_z["VIXCLS_z"]
    spx["cnsm"] = df_z["UMCSENT_z"]
    spx["m2"]   = df_z["M2SL_z"]
    spx["nfp"]  = df_z["PAYEMS_z"] 
    panels["SPX"] = _clip_universe(spx)

    # SP5MV 
    sp5mv = pd.DataFrame(index=df_z.index)
    sp5mv["vix"]   = df_z["VIXCLS_z"]
    sp5mv["rgdp"]  = df_z["A191RL1Q225SBEA_z"]
    sp5mv["cspd"]  = df_z["BAMLH0A0HYM2_z"]
    sp5mv["cnsm"]  = df_z["UMCSENT_z"]
    sp5mv["unmpr"] = df_z["UNRATE_z"] 
    panels["SP5MV"] = _clip_universe(sp5mv)

    return panels
