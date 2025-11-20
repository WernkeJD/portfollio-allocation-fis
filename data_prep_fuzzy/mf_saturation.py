import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from fis_arch import three_mf


def compute_mf_saturation(df: pd.DataFrame, input_name: str):
    """
    Given a DataFrame column (z-scored factor) and the input_name,
    compute how often Low / Mid / High is the dominant MF.
    """
    # Build a dummy Antecedent-like universe
    universe = np.linspace(-2.5, 2.5, 101)
    mfs = three_mf(input_name, type("V", (), {"universe": universe})())

    # Containers
    low_mf_vals = fuzz.interp_membership(universe, mfs[f"{input_name}_low"], df[input_name].values)
    mid_mf_vals = fuzz.interp_membership(universe, mfs[f"{input_name}_mid"], df[input_name].values)
    high_mf_vals = fuzz.interp_membership(universe, mfs[f"{input_name}_high"], df[input_name].values)

    # Determine which MF wins each time, winner = most domininant
    mf_stack = np.vstack([low_mf_vals, mid_mf_vals, high_mf_vals])
    winners = np.argmax(mf_stack, axis=0)

    # Count wins
    low_frac = np.mean(winners == 0)
    mid_frac = np.mean(winners == 1)
    high_frac = np.mean(winners == 2)

    return {
        "low": low_frac,
        "mid": mid_frac,
        "high": high_frac,
    }


def plot_mf_heatmap(factor_panels: dict):
    """Compute and plot MF saturation for every ETF and every factor."""
    for etf, df in factor_panels.items():
        rows = ["low", "mid", "high"]
        cols = list(df.columns)

        mat = np.zeros((3, len(cols)))

        for j, col in enumerate(cols):
            s = compute_mf_saturation(df, col)
            mat[0, j] = s["low"]
            mat[1, j] = s["mid"]
            mat[2, j] = s["high"]

        plt.figure(figsize=(12, 4))
        plt.imshow(mat, cmap="viridis", aspect="auto")
        plt.title(f"MF Saturation Heatmap for {etf}")
        plt.yticks(ticks=range(3), labels=rows)
        plt.xticks(ticks=range(len(cols)), labels=cols, rotation=45)
        plt.colorbar(label="Frequency")
        plt.tight_layout()
        plt.show()
