"""
All internet fetches & BeautifulSoup parsing live here.
Keeps engine.py perfectly pure – easier to unit-test offline.
"""
from __future__ import annotations
import re, io, base64, requests, warnings
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment
from .io import pace_df
from .constants import POWER_CONFS, COUNTING_STATS, DEFAULT_METRIC

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# -------------------------------------------------------------------
# ⬇️ 1.  put here: extract_height_weight_from_soup(...)
# ⬇️ 2.  put here: candidate_nbadraft_slugs(), fetch_nbadraft_ratings()
# ⬇️ 3.  put here: get_cbb_stats_from_url(), extract_position_from_url()
# Everything is an *exact copy* of the functions in your existing script.
# -------------------------------------------------------------------
