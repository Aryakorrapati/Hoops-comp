"""
Pure-math helpers (no network, no matplotlib, no file writes).
"""
import numpy as np
import pandas as pd
from math import erf as _erf
from .constants import DERIVED_COLS, COUNTING_STATS

# NumPy < 1.3 compatibility shim
if not hasattr(np, "erf"):
    np.erf = np.vectorize(_erf)

# ------------------------------------------------------------------
def dice_sim_matrix(v1: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Vectorised Sørensen–Dice similarity (higher = more similar)."""
    numer  = 2.0 * np.minimum(v1, M).sum(axis=1)
    denom  = v1.sum() + M.sum(axis=1)
    denom[denom == 0] = 1e-12
    return numer / denom

# ------------------------------------------------------------------
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-creates all derived metrics exactly like your notebook/one-file script.
    Paste your original body below the 3-line header.
    """
    out = df.copy()

    # ===--- existing code from your script starts here ---=================
    # (AST_to_TOV, 3PA_rate, Stocks40, TS, USG40, CreationLoad,
    #  AST_per_FGA, FT_rate, SelfCreationIdx …)
    # ----------------------------------------------------------------------
    # ✂️  <PASTE the full function body you already wrote> ✂️
    # ----------------------------------------------------------------------

    return out
