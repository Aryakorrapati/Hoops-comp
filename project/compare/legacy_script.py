import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import requests
from bs4 import BeautifulSoup
import re
import warnings
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from math import erf as _erf
import sys, re
from bs4 import Comment
import random, time
import requests
import cloudscraper

if not hasattr(np, "erf"):            # very old NumPy
    np.erf = np.vectorize(_erf)
elif not callable(getattr(np, "erf")):   # someone shadow‑patched it
    np.erf = np.vectorize(_erf)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

pace_df   = pd.read_csv("pace.csv")        
pace_long = pace_df.melt(id_vars=["Team"],
                         var_name="Season",
                         value_name="Pace")

PACE_LOOKUP = {                                   # key → pace
    (row.Team.strip().lower(), row.Season): float(row.Pace)
    for _, row in pace_long.iterrows()
}

def is_header_duplicate(row):
    # True if the row has identical Season/Team or the Team column looks like a season
    season_str = str(row['Season']).strip()
    team_str   = str(row['Team']).strip()
    return (
        team_str.lower() in {'team', 'career', 'total', 'summary', ''} or
        team_str == season_str or                       # e.g. '2015-16' in Team
        season_str.lower() == 'season'                  # literal header
    )

DERIVED_COLS = [
    'AST_to_TOV',      # already there
    '3PA_rate',
    'Stocks40',
    'TS',              # NEW ↓↓↓
    'USG40',
    'CreationLoad',
    'AST_per_FGA',
    'FT_rate',
    'SelfCreationIdx'  # (SCI)
]
BASE_NUMERIC = ['AST', 'TOV', '3PA', 'FGA', 'STL', 'BLK', 'MP']
COUNTING_STATS = [
    'PTS','TRB','AST','STL','BLK','ORB','DRB',
    'FGA','FGM','3P','3PA','FTA','FT','TOV','PF','2P','2PA'
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- safety helpers ---------------------------------------------------
    fga = out['FGA'].replace(0, np.nan)
    fta = out['FTA'].replace(0, np.nan)
    tov = out['TOV'].replace(0, np.nan)
    ast = out['AST'].replace(0, np.nan)
    mp  = out['MP'].replace(0, np.nan)

    # A. **Existing three**
    out['AST_to_TOV'] = ast / tov
    out['3PA_rate']   = out['3PA'] / fga
    out['Stocks40']   = (out['STL'] + out['BLK']) / mp * 40

    # B. **NEW six**
    denom_ts = 2.0 * (fga + 0.44 * fta)
    out['TS']      = out['PTS'] / denom_ts

    out['USG40']   = (fga + 0.44 * fta + tov) * 40 / mp               # simple proxy
    out['CreationLoad'] = (fga + 0.44 * fta + ast + tov) / mp

    out['AST_per_FGA'] = ast / fga
    out['FT_rate']     = fta / fga

    # ---- Self‑Creation Index (0–1 scale) ---------------------------------
    # ① normalise the three ingredients inside the dataframe
    cols_for_sci = ['CreationLoad', 'AST_per_FGA', 'FT_rate']
    for c in cols_for_sci:
        col = out[c]
        out[c + '_z'] = (col - col.mean()) / (col.std(ddof=0) + 1e-9)  # z‑score
        # rescale to 0‑1 via CDF of N(0,1)
        out[c + '_01'] = 0.5 * (1 + np.erf(out[c + '_z'] / np.sqrt(2)))

    out['SelfCreationIdx'] = (
        0.50 * out['CreationLoad_01'] +
        0.30 * out['AST_per_FGA_01'] +
        0.20 * out['FT_rate_01']
    )
    # ----------------------------------------------------------------------

    # Replace any inf / NaN that slipped through
    out[DERIVED_COLS] = out[DERIVED_COLS].fillna(0).replace([np.inf, -np.inf], 0)

    # Drop helper columns
    out.drop(columns=[c for c in out.columns if c.endswith(('_z', '_01'))], inplace=True)

    return out

def get_last_season_best_class_row(per_game):
    season_mask = per_game['Season'].astype(str).str.match(r'^\d{4}-\d{2}$')
    team_not_blank = per_game['Team'].notna() & per_game['Team'].astype(str).str.strip().ne('')
    not_header = ~per_game.apply(is_header_duplicate, axis=1)

    valid_rows = per_game[season_mask & team_not_blank & not_header]

    if valid_rows.empty:
        raise ValueError("No real season rows found")

    return valid_rows.iloc[-1]



def get_age_fact_from_url(cbb_url):
    tables = pd.read_html(cbb_url)
    per_game = None
    for t in tables:
        if 'Season' in t.columns and 'PTS' in t.columns and 'Team' in t.columns:
            per_game = t
            break
    if per_game is None:
        return 1

    last_row = get_last_season_best_class_row(per_game)
    class_val = str(last_row.get('Class', '')).strip().lower()
    if class_val.startswith('fr'):
        return 1
    elif class_val.startswith('so'):
        return 2
    else:
        return 3

def dice_sim_matrix(v1: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    v1 : (1, N) vector                     – your input player
    M  : (K, N) matrix of candidate rows   – players to compare
    returns: (K,) Dice similarity scores   – higher == more similar
    """
    # element‑wise min over axis 1  → shape (K,)
    numer   = 2.0 * np.minimum(v1, M).sum(axis=1)
    denom   = v1.sum() + M.sum(axis=1)
    # avoid div‑by‑zero
    denom[denom == 0] = 1e-12
    return numer / denom

def filter_by_physical_ratings(cbb: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """
    Return only those rows (players) whose Athleticism, Strength, and Quickness
    are each within ±1 (absolute difference ≤1) of the target player's values.

    Excludes the target player's own rows (unless you want to keep them).
    """

    required_cols = ["Player", "Athleticism", "Strength", "Quickness"]
    missing = [c for c in required_cols if c not in cbb.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Pick the target *reference* row: latest season by start year
    target_rows = cbb[cbb["Player"] == target_name]
    if target_rows.empty:
        raise ValueError(f"Target player '{target_name}' not found in dataset.")

    # Convert Season -> start year for sorting (robust to formats like 2021-22)
    def _start_year(s):
        m = re.match(r"^\s*(\d{4})", str(s))
        return int(m.group(1)) if m else -1

    target_rows = target_rows.copy()
    target_rows["__start_year"] = target_rows["Season"].map(_start_year)
    target_row = target_rows.sort_values("__start_year").iloc[-1]

    t_ath = target_row["Athleticism"]
    t_str = target_row["Strength"]
    t_quk = target_row["Quickness"]

    # Safety: ensure numeric
    for col in ["Athleticism", "Strength", "Quickness"]:
        cbb[col] = pd.to_numeric(cbb[col], errors="coerce")

    mask = (
        (cbb["Athleticism"].sub(t_ath).abs() <= 1) &
        (cbb["Strength"].sub(t_str).abs()     <= 1) &
        (cbb["Quickness"].sub(t_quk).abs()    <= 1) &
        (cbb["Player"] != target_name)  # drop self
    )
    return cbb[mask]

def get_cbb_stats_from_url(cbb_url, stat_cols):
    r = requests.get(cbb_url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')

    # --- Height/weight extraction ---
    def extract_height_weight_from_soup(soup):
        patterns = [
            re.compile(r'(\d+)-(\d+),\s*(\d+)lb'),
            re.compile(r'(\d+)cm,?\s*(\d+)kg'),
        ]
        meta_div = soup.find('div', id='meta') or soup.find('div', id='info')
        search_texts = []
        if meta_div:
            for tag in meta_div.find_all(['p', 'span', 'strong']):
                t = tag.get_text(" ", strip=True)
                if t:
                    search_texts.append(t)
            search_texts.append(meta_div.get_text(" ", strip=True))
        for search_text in search_texts:
            for pat in patterns:
                m = pat.search(search_text)
                if m:
                    if len(m.groups()) == 3:
                        feet, inches, lbs = int(m.group(1)), int(m.group(2)), int(m.group(3))
                        return feet * 12 + inches, lbs
                    if len(m.groups()) == 2:
                        cm, kg = int(m.group(1)), int(m.group(2))
                        return round(cm / 2.54), round(kg * 2.205)
        return None, None

    POWER_CONFS = {
    "SEC": ["SEC", "Southeastern Conference"],
    "ACC": ["ACC", "Atlantic Coast Conference"],
    "Big Ten": ["Big Ten", "B1G", "Big Ten Conference"],
    "Big 12": ["Big 12", "Big 12 Conference"],
    "Pac-12": ["Pac-12", "Pacific-12 Conference"],
    "Big East": ["Big East", "Big East Conference"]
    }

    tables = pd.read_html(cbb_url)
    per_game = None
    for t in tables:
        if 'Season' in t.columns and 'PTS' in t.columns and 'Team' in t.columns:
            per_game = t
            break
    if per_game is None:
        raise ValueError("No per-game table found at " + cbb_url)

    last_row = get_last_season_best_class_row(per_game)
    season_str  = str(last_row['Season'])
    first_year  = int(season_str[:4])       
    if first_year < 2004:
        raise ValueError(
            f"{season_str} is earlier than 2004-05. Cannot use this player"
        )

    conf_raw   = last_row['Conf']
    conf_clean = str(conf_raw).strip().upper()
    POWER_CONFS = {'ACC','SEC','BIG TEN','BIG 12','BIG EAST','PAC-12','PAC-10'}
    conf_power = 7 if conf_clean in POWER_CONFS else 1

    cls = str(last_row['Class']).strip().lower()
    if cls.startswith('fr'):
        age_fact = 1
    elif cls.startswith('so'):
        age_fact = 2
    else:
        age_fact = 3


    # Height and weight from HTML
    height_in, weight_lb = extract_height_weight_from_soup(soup)

    # ------- per-100 conversion for the player --------------------------
    team_key   = str(last_row['Team']).strip().lower()
    season_key = str(last_row['Season']).strip()
    pace_val   = PACE_LOOKUP.get((team_key, season_key))

    main_stats = [                     # stays the same list comprehension
        float(last_row[col]) if col in last_row and pd.notnull(last_row[col]) else 0.0
        for col in stat_cols[:-3]
    ]

    if pace_val and pace_val > 0:
        per100_factor = 100.0 / pace_val
        for i, col in enumerate(stat_cols[:-3]):
            if col in COUNTING_STATS:          # scale counts only
                main_stats[i] *= per100_factor
    main_stats.append(float(height_in) if height_in is not None else 0.0)
    main_stats.append(float(weight_lb) if weight_lb is not None else 0.0)
    main_stats.append(float(conf_power))

    player_name_tag = soup.find('h1')
    player_name = player_name_tag.text.strip() if player_name_tag else "Input Player"
    return (
        np.array(main_stats).reshape(1, -1),  # vector for similarity
        last_row['Season'],                   # season label
        player_name,                          # player name
        age_fact,                             # 1 / 2 / 3
        last_row                              # ← NEW: raw per‑game Series
    )

def extract_position_from_url(cbb_url):
    r = requests.get(cbb_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    meta_div = soup.find('div', id='meta') or soup.find('div', id='info')
    if not meta_div:
        return None

    for p_tag in meta_div.find_all('p'):
        strong = p_tag.find('strong')
        if strong and 'position' in strong.text.strip().lower():
            sibling_text = strong.next_sibling
            if sibling_text:
                text = sibling_text.strip().lower()
                if text.startswith('guard'):
                    return 'g'
                elif text.startswith('forward'):
                    return 'f'
                elif text.startswith('center'):
                    return 'c'
    return None


# --- Load CBB stats ---
cbb_df = pd.read_csv('cbb_stats.csv')

for col in ['ortg', 'drtg', 'ws/40']:
    if col not in cbb_df.columns:
        cbb_df[col] = np.nan
    cbb_df[col] = pd.to_numeric(cbb_df[col], errors='coerce')

# 1) make sure there is a 'Team' column
if "Team" not in cbb_df.columns:
    for alt in ["School", "Tm", "College"]:
        if alt in cbb_df.columns:
            cbb_df = cbb_df.rename(columns={alt: "Team"})
            break
    else:                       # runs if the loop *didn't* break
        raise ValueError(
            "cbb_stats.csv has no Team/School column – check the header!"
        )

# 2) make sure there is a 'Season' column (rarely called 'Year')
if "Season" not in cbb_df.columns:
    if "Year" in cbb_df.columns:
        cbb_df = cbb_df.rename(columns={"Year": "Season"})
    else:
        raise ValueError(
            "cbb_stats.csv has no Season column – check the header!"
        )
# --- 3a. helper: per-100 conversion --------------------------------------
def to_per100(df):
    df = df.copy()
    for idx, row in df.iterrows():
        key  = (row['Team'].strip().lower(), row['Season'])
        pace = PACE_LOOKUP.get(key)            # pace = possessions / game
        if pace and pace > 0:
            factor = 100.0 / pace
            for col in COUNTING_STATS:
                if col in df.columns:
                    df.at[idx, col] = df.at[idx, col] * factor
    return df

for _col in ("Athleticism", "Strength", "Quickness"):
    if _col not in cbb_df.columns:
        raise ValueError(f"Missing column '{_col}' in cbb_stats.csv (populate it first).")


# --- 3b. keep TWO versions ----------------------------------------------
cbb_df_raw     = cbb_df.copy()      # <- untouched (for display later)
cbb_df_per100  = to_per100(cbb_df)  # <- goes into the model
cbb_df_ext     = add_derived_features(cbb_df_per100)
non_stat_cols = ['Player', 'Season', 'Year', 'Team', 'Explosiveness']

# start from every column that isn't one of the non‑stat / meta columns
base_stat_cols = [
    c for c in cbb_df.columns
    if c not in non_stat_cols + ['Height_in', 'Weight_lb', 'Conf_Power', 'age_fact', 'Position']
]

# ensure the three new metrics are present (if they exist in the file)
metric_cols = [c for c in ['ortg', 'drtg', 'ws/40'] if c in cbb_df.columns]

# build final stat_cols in this order:
#   base stats ... + (ortg, drtg, ws/40) + Height_in + Weight_lb + Conf_Power
# if any of the metric_cols were already in base_stat_cols, we'll de‑duplicate
stat_cols = []
seen = set()
for c in base_stat_cols + metric_cols:
    if c not in seen:
        stat_cols.append(c); seen.add(c)

for extra in ['ortg', 'drtg', 'ws/40']:
    if extra in cbb_df.columns and extra not in stat_cols:
        stat_cols.append(extra)

for extra in ['Height_in', 'Weight_lb', 'Conf_Power']:
    if extra not in stat_cols:
        stat_cols.append(extra)

all_cols = stat_cols + DERIVED_COLS     
for extra in ['Height_in', 'Weight_lb', 'Conf_Power']:
    if extra not in stat_cols:
        stat_cols += [extra]


DEFAULT_METRIC = 100.0  # fallback if not found

def _norm_season(s):
    s = str(s).strip()
    m = re.match(r'^(\d{4}-\d{2})', s)
    return m.group(1) if m else None

def get_input_metrics_from_csv(name_lower: str, season_str: str):
    """
    Look up (ortg, drtg, ws40) for the given player/season from cbb_df_raw.
    1) Try exact season match.
    2) If missing, fall back to the most recent earlier season with any value.
    3) If still missing, return DEFAULT_METRIC for each.
    """
    target = _norm_season(season_str)
    rows = cbb_df_raw[cbb_df_raw['Player'].str.lower() == name_lower].copy()
    if rows.empty:
        return (DEFAULT_METRIC, DEFAULT_METRIC, DEFAULT_METRIC)

    # sortable start year
    def _start_year(s):
        m = re.match(r'^\s*(\d{4})', str(s))
        return int(m.group(1)) if m else -1

    rows['__start'] = rows['Season'].map(_start_year)

    # ---- 1) exact season match
    if target:
        mask = rows['Season'].astype(str).map(_norm_season) == target
        exact = rows[mask]
        if not exact.empty:
            r = exact.sort_values('__start').iloc[-1]
            o = pd.to_numeric(r.get('ortg'),  errors='coerce')
            d = pd.to_numeric(r.get('drtg'),  errors='coerce')
            w = pd.to_numeric(r.get('ws/40'), errors='coerce')
            o = float(o) if pd.notna(o) else None
            d = float(d) if pd.notna(d) else None
            w = float(w) if pd.notna(w) else None
            if o is not None or d is not None or w is not None:
                return (o if o is not None else DEFAULT_METRIC,
                        d if d is not None else DEFAULT_METRIC,
                        w if w is not None else DEFAULT_METRIC)

    # ---- 2) fallback: most recent season with any value
    for _, r in rows.sort_values('__start').iloc[::-1].iterrows():
        o = pd.to_numeric(r.get('ortg'),  errors='coerce')
        d = pd.to_numeric(r.get('drtg'),  errors='coerce')
        w = pd.to_numeric(r.get('ws/40'), errors='coerce')
        if pd.notna(o) or pd.notna(d) or pd.notna(w):
            o = float(o) if pd.notna(o) else DEFAULT_METRIC
            d = float(d) if pd.notna(d) else DEFAULT_METRIC
            w = float(w) if pd.notna(w) else DEFAULT_METRIC
            return (o, d, w)

    # ---- 3) nothing found
    return (DEFAULT_METRIC, DEFAULT_METRIC, DEFAULT_METRIC)


# -- Get your input vector from a pasted CBB player URL --
input_cbb_url = input("Paste a CBB Sports Reference player URL: ").strip()

if not re.match(
        r'^https?://(?:www\.)?sports-reference\.com/cbb/players/[^/]+\.html$',
        input_cbb_url):
    print("The link must be a College-Basketball Sports-Reference player page.")
    sys.exit(1)

try:
    (input_vector, input_season, input_player_name,
     input_age_fact, last_row_raw) = get_cbb_stats_from_url(input_cbb_url, stat_cols)
except ValueError as err:       # catches “earlier than 1999‑00” and any other
    print(f"{err}")          # just a clean one‑line notice
    import sys
    sys.exit(0)                 # stop the program quietly
input_age_fact = get_age_fact_from_url(input_cbb_url)

def safe_float(x):
    """Return float(x) or None if not parseable / NaN."""
    try:
        # handle pandas NaN
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        s = str(x).replace(",", "").strip()
        return float(s)
    except Exception:
        return None

def _season_norm(s):
    s = str(s).strip()
    m = re.match(r"^(\d{4}-\d{2})", s)
    return m.group(1) if m else None

def _unwrap_comments(html: str, wanted=("players_per_poss", "players_advanced")) -> str:
    """
    Sports-Reference hides some tables in HTML comments. This replaces those
    comment nodes with their parsed HTML so pandas can see the tables.
    """
    soup = BeautifulSoup(html, "html.parser")
    inserted = 0
    for node in soup.find_all(string=lambda t: isinstance(t, Comment)):
        frag = str(node)
        if any(w in frag for w in wanted):
            node.replace_with(BeautifulSoup(frag, "html.parser"))
            inserted += 1
    # print(f"[TRACE] uncomment fragments={inserted}")
    return str(soup)

def scrape_input_metrics_from_url(cbb_url: str, season_str: str):
    """
    Scrape ORtg, DRtg from per-possession table (players_per_poss)
    and WS/40 from advanced table (players_advanced) for the given season.

    Returns: (ortg, drtg, ws40_scaled) where ws40_scaled = WS/40 * 1000.
    Missing values return as None.
    """
    try:
        r = requests.get(cbb_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"[WARN] Fetch failed for {cbb_url}: {e}")
        return (None, None, None)

    html = _unwrap_comments(r.text)
    soup = BeautifulSoup(html, "html.parser")

    def _read_table(table_id):
        tag = soup.find("table", id=table_id)
        if not tag:
            return None
        try:
            dfs = pd.read_html(str(tag))
        except ValueError:
            return None
        if not dfs:
            return None
        df = dfs[0]
        # flatten potential MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            flat = []
            for tup in df.columns:
                parts = [str(x).strip() for x in tup if str(x).strip() and not str(x).startswith("Unnamed")]
                flat.append(parts[-1] if parts else str(tup[-1]))
            df.columns = flat
        else:
            df.columns = [str(c).strip() for c in df.columns]
        return df

    target = _season_norm(season_str)
    ortg = drtg = ws40_scaled = None

    # Per 100 poss
    per_poss = _read_table("players_per_poss")
    if per_poss is not None and "Season" in per_poss.columns:
        mask = per_poss["Season"].astype(str).map(_season_norm) == target
        rows = per_poss[mask]
        if not rows.empty:
            rlast = rows.iloc[-1]
            ortg = safe_float(rlast.get("ORtg"))
            drtg = safe_float(rlast.get("DRtg"))

    # Advanced
    adv = _read_table("players_advanced")
    if adv is not None and "Season" in adv.columns:
        mask = adv["Season"].astype(str).map(_season_norm) == target
        rows = adv[mask]
        if not rows.empty:
            rlast = rows.iloc[-1]
            # column name might be exactly "WS/40"
            ws_col = next((c for c in rows.columns if c.upper().startswith("WS/40")), None)
            if ws_col:
                ws = safe_float(rlast.get(ws_col))
                if ws is not None:
                    ws40_scaled = ws * 1000.0
            # fallback for ORtg/DRtg if still None
            if ortg is None:
                ortg = safe_float(rlast.get("ORtg"))
            if drtg is None:
                drtg = safe_float(rlast.get("DRtg"))

    print(f"[LIVE] {cbb_url.split('/')[-1]} {_season_norm(season_str)} ORtg={ortg} DRtg={drtg} WS40={ws40_scaled}")
    return (ortg, drtg, ws40_scaled)

input_player_name_lower = input_player_name.strip().lower()
o_val, d_val, w_val = get_input_metrics_from_csv(input_player_name_lower, str(input_season))

_dbg_rows = cbb_df_raw[cbb_df_raw['Player'].str.lower() == input_player_name_lower].copy()
print("\n[DEBUG] CSV rows for input player:", input_player_name)
print(_dbg_rows[['Player','Season','ortg','drtg','ws/40']].to_string(index=False))

print(f"[DEBUG] Selected season={input_season}  ->  o={o_val}  d={d_val}  w={w_val}")

if (o_val == DEFAULT_METRIC and d_val == DEFAULT_METRIC and w_val == DEFAULT_METRIC):
    o2, d2, w2 = scrape_input_metrics_from_url(input_cbb_url, str(input_season))
    if o2 is not None: o_val = float(o2)
    if d2 is not None: d_val = float(d2)
    if w2 is not None: w_val = float(w2)

print(f"[DEBUG-FINAL-METRICS] ORtg={o_val} DRtg={d_val} WS40={w_val}")

# Also confirm the vector actually received these values:
def _idx(col):
    return stat_cols.index(col) if col in stat_cols else None

for col,val in [('ortg',o_val),('drtg',d_val),('ws/40',w_val)]:
    i = _idx(col)
    if i is not None:
        print(f"[DEBUG] input_vector[{col}] index={i} value_in_vector={input_vector[0,i]}")
    else:
        print(f"[DEBUG] {col} not in stat_cols -> will never appear in vectors!")

def _safe_set(col, val):
    if col in stat_cols:
        idx = stat_cols.index(col)
        input_vector[0, idx] = float(val)

_safe_set('ortg',  o_val)
_safe_set('drtg',  d_val)
_safe_set('ws/40', w_val)

input_cbb_dict = dict(zip(stat_cols, input_vector.flatten()))

input_height = input_vector[0, -3]  # Second to last in the stat vector
input_weight = input_vector[0, -2]  # Last in the stat vector

ast  = float(input_cbb_dict['AST'])
tov  = float(input_cbb_dict['TOV'])
trey = float(input_cbb_dict['3PA'])
fga  = float(input_cbb_dict['FGA'])
stl  = float(input_cbb_dict['STL'])
blk  = float(input_cbb_dict['BLK'])
mp   = float(input_cbb_dict['MP'])
pts  = float(input_cbb_dict['PTS'])
fta  = float(input_cbb_dict['FTA'])

cbb_positions = cbb_df['Position'].values 
input_position = extract_position_from_url(input_cbb_url)

player_names = cbb_df['Player'].values
player_vectors = cbb_df_ext[all_cols].values.astype(float)


# --- HEIGHT/WEIGHT FILTER APPLIED TO COMP SELECTION ---
# For CBB comp (looser, since you want similar style, not just size)
cbb_height_tol = 2  # for example
cbb_weight_tol = 30

# For NBA floor/ceiling comps (stricter for true size comps)
nba_height_tol = 1
nba_weight_tol = 20

RATING_TOL = 1   # ±1 window for Ath/Str/Quickness
MISSING_FIELD_VALUE = 6      # default stand‑in when a rating is N/A
MISSING_FIELD_TOL   = 10     # relaxed window when that rating was N/A

# ---------- 2.  REBUILD ARRAYS AFTER TRIM --------------------------
cbb_player_vectors = cbb_df_ext[all_cols].values.astype(float)
cbb_player_names   = cbb_df_ext['Player'].values

# ---------- 2a.  DEFINE input_player_name_lower EARLY --------------
input_player_name_lower = input_player_name.strip().lower()

# ---------- NBADraft.net scraping helpers (Ath / Str / Quick) ----------------
NBADRAFT_FIELDS = ["Athleticism", "Strength", "Quickness"]

_suffix_tokens = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v", "vi", "vii"}

_NBADRAFT_OVERRIDES = {
    "vj edgecombe": "vj-edgecombe",
    "v j edgecombe": "vj-edgecombe",
    "v.j. edgecombe": "vj-edgecombe",
    "aj johnson": "aj-johnson",
    "a j johnson": "aj-johnson",
    "a.j. johnson": "aj-johnson",
    # add more if you encounter others
}

def _norm_name(n: str) -> str:
    return re.sub(r"\s+", " ", n.lower().replace("’", "'")).strip()

def candidate_nbadraft_slugs(name: str):
    """
    Yield likely slug candidates for NBADraft.net.
    Handles initials like 'V.J.' -> 'vj', suffix removal, and fallbacks.
    """
    norm = _norm_name(name)

    # 0) override
    if norm in _NBADRAFT_OVERRIDES:
        yield _NBADRAFT_OVERRIDES[norm]

    # 1) remove punctuation except spaces/hyphens
    tmp = re.sub(r"[^\w\s\-]", " ", norm)  # keep letters/digits/underscore
    tokens = [t for t in re.findall(r"[a-z0-9]+", tmp) if t not in _suffix_tokens]

    if not tokens:
        return

    # 2) plain join
    yield "-".join(tokens)

    # 3) if first 2 tokens are single letters -> combine them (V J -> vj)
    if len(tokens) >= 2 and len(tokens[0]) == 1 and len(tokens[1]) == 1:
        combined = ["".join(tokens[:2])] + tokens[2:]
        yield "-".join(combined)
        # also try with explicit hyphen between initials
        hyphenated = [tokens[0] + tokens[1]] + tokens[2:]
        yield "-".join(hyphenated)

    # 4) if first token is single letter and there are ≥2 more tokens, try drop/keep variants
    if len(tokens) >= 3 and len(tokens[0]) == 1:
        yield "-".join(tokens[1:])                    # drop the single initial
        yield "-".join([tokens[0] + tokens[1]] + tokens[2:])  # combine first two

HEADERS_LIST = [
    # rotate a few realistic desktop UA strings:
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
     "AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/126.0.0.0 Safari/537.36"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) "
     "AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "Version/17.0 Safari/605.1.15"),
]

def _get_html_cloudflare(url: str, ua: str) -> str | None:
    """Return page HTML or None on error/ban."""
    scraper = cloudscraper.create_scraper(
        browser={"custom": ua},
        delay=2,                 # polite pause when Cloudflare says “wait”
    )
    try:
        r = scraper.get(url, timeout=25)
        if r.status_code == 404:
            return None         # slug doesn't exist
        if r.status_code != 200:
            print(f"[WARN] NBADraft HTTP {r.status_code} at {url}")
            return None
        return r.text
    except Exception as exc:
        print(f"[WARN] NBADraft request failed: {exc}")
        return None

def fetch_nbadraft_ratings(player_name: str):
    NBADRAFT_FIELDS = ["Athleticism", "Strength", "Quickness"]
    base = {f: None for f in NBADRAFT_FIELDS}

    for slug in candidate_nbadraft_slugs(player_name):
        url = f"https://www.nbadraft.net/players/{slug}/"

        # 1️⃣  ── fetch the raw HTML (your existing code)
        try:
            resp = cloudscraper.create_scraper().get(url, timeout=20)
            if resp.status_code != 200:
                continue
        except Exception:
            continue
        html = resp.text            # ← this is what we’ll inspect

        # 2️⃣  ── DEBUG DUMP  (insert right here)
        if "rating-block" in html or "player-detail-table" in html:
            import re, textwrap
            snippet = re.search(r"Athleticism(.{0,120})", html, re.I)
            print(
                "[RAW]",
                textwrap.shorten(
                    snippet.group(0) if snippet else html[:300],
                    width=200,
                    placeholder=" … "
                )
            )

        import re, textwrap
        match = re.search(r"Athleticism(.{0,150})</tr>", html, re.I|re.S)
        print("[DBG-HTML]", textwrap.shorten(match.group(0) if match else html[:400], 250))
        
        soup = BeautifulSoup(html, "html.parser")
        ratings = dict(base)

        # ---- parse the table ----
        table = soup.find("table", class_="player-detail-table")
        if table:
            for row in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
                if len(cells) >= 2:
                    key = cells[0].strip().lower()
                    val = cells[1]
                    for fld in NBADRAFT_FIELDS:
                        if key.startswith(fld.lower()):
                            try:
                                ratings[fld] = int(val)
                            except ValueError:
                                pass

        missing = {f for f, v in ratings.items() if v is None}
        print(f"[INFO] NBADraft ratings for {player_name} ({slug}): "
              + ", ".join(f"{k}={ratings[k] or 'N/A'}" for k in NBADRAFT_FIELDS))
        return ratings, missing

    # fall-back when every slug blocked / missing
    print(f"[WARN] NBADraft page missing or blocked for {player_name}")
    return base, set(NBADRAFT_FIELDS)


# ---------- 2a.  DEFINE input_player_name_lower EARLY --------------
input_player_name_lower = input_player_name.strip().lower()

# ---------- 2b.  PHYSICAL RATING BASELINE WITH NBADraft FALLBACK ----
def _season_start_year__phys(s):
    m = re.match(r'^\s*(\d{4})', str(s))
    return int(m.group(1)) if m else -1

# Try exact full-match rows (player already in cbb_stats.csv)
_phys_rows = cbb_df[cbb_df['Player'].str.lower() == input_player_name_lower]

# Fallback: first + last token match (handles middle names, suffix removal)
if _phys_rows.empty:
    tokens = input_player_name_lower.split()
    if len(tokens) >= 2:
        first, last = tokens[0], tokens[-1]
        _phys_rows = cbb_df[
            cbb_df['Player'].str.lower().apply(
                lambda n: (lambda ts: len(ts) >= 2 and ts[0] == first and ts[-1] == last)(n.split())
            )
        ]

use_phys_filter = True

rating_tols = {
        "Athleticism": RATING_TOL,
        "Strength":    RATING_TOL,
        "Quickness":   RATING_TOL
    }

if _phys_rows.empty:
    # Not in dataset: scrape NBADraft.net
    scraped, missing = fetch_nbadraft_ratings(input_player_name)
    input_ath = scraped["Athleticism"]
    input_str = scraped["Strength"]
    input_quk = scraped["Quickness"]

    if missing:
        print(f"[INFO] Missing NBADraft fields: {', '.join(missing)} "
            f"-> substituting {MISSING_FIELD_VALUE} and widening tolerance.")
        if "Athleticism" in missing: input_ath = MISSING_FIELD_VALUE
        if "Strength"     in missing: input_str = MISSING_FIELD_VALUE
        if "Quickness"    in missing: input_quk = MISSING_FIELD_VALUE

    # If still None (page missing or ratings absent), disable filter
    if any(v is None for v in (input_ath, input_str, input_quk)):
        use_phys_filter = False
        input_ath = input_str = input_quk = None
else:
    # Player exists in dataset: take LAST (latest season) row
    _phys_rows = _phys_rows.copy()
    _phys_rows['__start_year'] = _phys_rows['Season'].map(_season_start_year__phys)
    _input_phys_row = _phys_rows.sort_values('__start_year').iloc[-1]

    input_ath = pd.to_numeric(_input_phys_row['Athleticism'], errors='coerce')
    input_str = pd.to_numeric(_input_phys_row['Strength'],    errors='coerce')
    input_quk = pd.to_numeric(_input_phys_row['Quickness'],   errors='coerce')

    if any(pd.isna([input_ath, input_str, input_quk])):
        # Dataset row exists but missing values; fallback to scrape
        print("[WARN] Dataset ratings incomplete; attempting NBADraft scrape.")
        scraped = fetch_nbadraft_ratings(input_player_name)
        for k, v in scraped.items():
            if k == "Athleticism" and pd.isna(input_ath) and v is not None: input_ath = v
            if k == "Strength"     and pd.isna(input_str) and v is not None: input_str = v
            if k == "Quickness"    and pd.isna(input_quk) and v is not None: input_quk = v

        if any(pd.isna([input_ath, input_str, input_quk])):
            print("[INFO] Still missing one or more ratings -> disabling physical rating filter.")
            use_phys_filter = False
            input_ath = input_str = input_quk = None

# Age window allowed for comps (depends on class)
if input_age_fact == 1:
    allowed_ages = [1, 2]
elif input_age_fact == 2:
    allowed_ages = [1, 2, 3]
else:
    allowed_ages = [2, 3]
# -------------------------------------------------------------------


if use_phys_filter:
    phys_mask = (
        (cbb_df_ext['Athleticism'].sub(input_ath).abs() <= rating_tols["Athleticism"]) &
        (cbb_df_ext['Strength'].sub(input_str).abs()     <= rating_tols["Strength"]) &
        (cbb_df_ext['Quickness'].sub(input_quk).abs()    <= rating_tols["Quickness"])
    )
else:
    phys_mask = np.ones(len(cbb_df_ext), dtype=bool)   # everyone passes

cbb_size_mask = (
      phys_mask
    & (np.abs(cbb_df_ext['Height_in'].values  - input_height) <= cbb_height_tol)
    & (np.abs(cbb_df_ext['Weight_lb'].values - input_weight)  <= cbb_weight_tol)
    & (cbb_df_ext['age_fact'].astype(int).isin(allowed_ages))
    & (cbb_df_ext['Position'].str.lower() == input_position)
)

cbb_player_vectors = cbb_df_ext[all_cols].values.astype(float)   # ensure defined earlier
cbb_player_names   = cbb_df_ext['Player'].values

filtered_cbb_vectors = cbb_player_vectors[cbb_size_mask]
filtered_cbb_names   = cbb_player_names[cbb_size_mask]



# ==================================================================


is_not_input = np.fromiter(
    (name.strip().lower() != input_player_name_lower
     for name in filtered_cbb_names),
    dtype=bool,
    count=len(filtered_cbb_names)          # guarantees bool dtype even if len==0
)

filtered_cbb_names   = filtered_cbb_names[is_not_input]
filtered_cbb_vectors = filtered_cbb_vectors[is_not_input]

if len(filtered_cbb_names) == 0:
    raise ValueError(
        "No CBB players found meeting physical (±1 Ath/Str/Qk) + size/age/position/explosiveness filters. "
        "Consider loosening one of the thresholds."
    )

# ------- new metrics ----------
ast_to_tov = ast / tov if tov else 0
trey_rate  = float(input_cbb_dict['3PA']) / fga if fga else 0
stocks40   = (stl + blk) / mp * 40 if mp else 0
ts         = pts / (2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) else 0
usg40      = (fga + 0.44 * fta + tov) * 40 / mp if mp else 0
cre_load   = (fga + 0.44 * fta + ast + tov) / mp if mp else 0
ast_per_fga= ast / fga if fga else 0
ft_rate    = fta / fga if fga else 0
# SCI will be scaled after we know dataset means/stdev – use 0 placeholder here
sci_placeholder = 0.0

input_derived = np.array([
    ast_to_tov,          # AST_to_TOV
    trey_rate,           # 3PA_rate
    stocks40,            # Stocks40
    ts,                  # TS
    usg40,               # USG40
    cre_load,            # CreationLoad
    ast_per_fga,         # AST_per_FGA
    ft_rate,             # FT_rate
    sci_placeholder      # SelfCreationIdx – we’ll overwrite below
], dtype=float)

input_vector_full = np.concatenate([input_vector.flatten(), input_derived])
input_vector      = input_vector_full.reshape(1, -1)

# ---------------- now scale & insert Self‑Creation Idx --------------------
sci_z = (
    0.50 * (cre_load   - cbb_df_ext['CreationLoad'].mean()) / cbb_df_ext['CreationLoad'].std(ddof=0) +
    0.30 * (ast_per_fga- cbb_df_ext['AST_per_FGA'].mean())  / cbb_df_ext['AST_per_FGA'].std(ddof=0) +
    0.20 * (ft_rate    - cbb_df_ext['FT_rate'].mean())      / cbb_df_ext['FT_rate'].std(ddof=0)
)
sci_input = 0.5 * (1 + np.erf(sci_z / np.sqrt(2)))
sci_min   = cbb_df_ext['SelfCreationIdx'].min()
sci_max   = cbb_df_ext['SelfCreationIdx'].max()
sci_input = np.clip((sci_input - sci_min) / (sci_max - sci_min + 1e-9), 0, 1)

input_vector[0, -1] = sci_input           # overwrite placeholder

def _set_metric(col, val):
    if col in stat_cols:
        j = stat_cols.index(col)
        input_vector[0, j] = float(val)

# We already computed these earlier:
# input_player_name_lower = input_player_name.strip().lower()
# o_val, d_val, w_val = get_input_metrics_from_csv(input_player_name_lower, str(input_season))

_set_metric('ortg',  o_val)
_set_metric('drtg',  d_val)
_set_metric('ws/40', w_val)

# Debug to verify final vector contains them
for col in ('ortg','drtg','ws/40'):
    if col in stat_cols:
        j = stat_cols.index(col)
        print(f"[DEBUG‑FINAL] vector[{col}] idx={j} val={input_vector[0, j]}")


# --- Cosine (higher = better) -------------------------------------
sim_cos  = cosine_similarity(input_vector, filtered_cbb_vectors)[0]

# --- Scaled Euclidean (higher = better after inversion) -----------
dist_raw = euclidean_distances(input_vector, filtered_cbb_vectors)[0]
inv_scaled_dist = 1 - (dist_raw - dist_raw.min()) / (dist_raw.ptp() + 1e-9)

# --- Dice (higher = better) ---------------------------------------
sim_dice = dice_sim_matrix(input_vector, filtered_cbb_vectors)

# --- Combine the three --------------------------------------------
# You can tweak these weights; start with equal importance for cosine & dice,
# then let Euclidean fine‑tune the ranking.
w_cos  = 0.3
w_dice = 0.3
w_euc  = 0.4

combined_score = (
      w_cos  * sim_cos
    + w_dice * sim_dice
    + w_euc  * inv_scaled_dist
)

similarities = sim_cos      # old variable expected later
distances    = dist_raw     # old variable expected later

order          = np.argsort(-combined_score)        # best ➜ worst
ranked_names   = filtered_cbb_names[order]
ranked_vectors = filtered_cbb_vectors[order]
ranked_scores  = combined_score[order]
ranked_dists   = distances[order]

nba_df = pd.read_csv('nba_stats.csv')
nba_ppg_map = (nba_df.set_index('Player')['PTS'].astype(float).to_dict())

show_good = False          # placeholder; will be updated later
good_pct  = 0.0            # avoids "referenced before assignment"

good_idx = next(
    (i for i, name in enumerate(ranked_names)
     if nba_ppg_map.get(name, 0) >= 10),
    None
)

if good_idx is None:
    good_idx = 0            # fallback to ordinary comp

def dice_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    numerator   = 2.0 * np.minimum(vec_a, vec_b).sum()
    denominator = vec_a.sum() + vec_b.sum()
    return 0.0 if denominator == 0 else numerator / denominator


best_idx = 0                        # first in ranked list
top_idx  = best_idx                 # keep variable name for later code

# ordinary (possibly poor) comp
best_cbb_player     = ranked_names[best_idx]
best_cbb_similarity = similarities[ order[best_idx] ]
best_cbb_distance   = ranked_dists[best_idx]
best_cbb_combined   = ranked_scores[best_idx]
best_vec_full       = ranked_vectors[best_idx]

# good player comp (≥10 PPG)
good_cbb_player     = ranked_names[good_idx]
good_vec_full       = ranked_vectors[good_idx]
good_cbb_similarity = similarities[ order[good_idx] ]
good_cbb_distance   = ranked_dists[good_idx]
good_cbb_combined   = ranked_scores[good_idx]

dice_score_raw = dice_similarity(input_vector[0], best_vec_full)
dice_score_pct = dice_score_raw*100

# --- NBA stats ---
nba_non_stat_cols = ['Player', 'Season']
nba_stat_cols = [c for c in nba_df.columns if c not in nba_non_stat_cols]
if 'Height_in' not in nba_stat_cols:
    nba_stat_cols += ['Height_in']
if 'Weight_lb' not in nba_stat_cols:
    nba_stat_cols += ['Weight_lb']
output_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK']
output_indices = [nba_stat_cols.index(s) for s in output_stats]

nba_vectors = nba_df[nba_stat_cols].values.astype(float)
nba_names = nba_df['Player'].values

nba_nan_mask = np.isnan(nba_vectors).any(axis=1)
nba_vectors = nba_vectors[~nba_nan_mask]
nba_names = nba_names[~nba_nan_mask]

# Only allow candidates within tolerance of input player's height/weight
nba_comp_row = nba_df[nba_df['Player'].str.lower() == best_cbb_player.lower()]
if nba_comp_row.empty:
    raise ValueError(f"No NBA stats found for CBB comp: {best_cbb_player}")
nba_comp_vector = nba_comp_row[nba_stat_cols].values.astype(float)[0].reshape(1, -1)
nba_comp_height = nba_comp_row['Height_in'].values[0]
nba_comp_weight = nba_comp_row['Weight_lb'].values[0]

within_tol = [
    idx for idx in range(nba_vectors.shape[0])
    if abs(nba_vectors[idx, nba_stat_cols.index('Height_in')] - nba_comp_height) <= nba_height_tol and
       abs(nba_vectors[idx, nba_stat_cols.index('Weight_lb')] - nba_comp_weight) <= nba_weight_tol
]

filtered_nba_vectors = nba_vectors[within_tol]
filtered_nba_names = nba_names[within_tol]

nba_similarities = cosine_similarity(nba_comp_vector, filtered_nba_vectors)[0]
nba_comp_idx_in_filtered = np.argmax(nba_similarities)
candidate_indices = [
    i for i in range(len(filtered_nba_names))
    if i != nba_comp_idx_in_filtered            # was already there
    and filtered_nba_names[i].strip().lower() != input_player_name_lower
]

if len(candidate_indices) == 0:
    nba_floor_player = filtered_nba_names[nba_comp_idx_in_filtered]
    nba_ceiling_player = filtered_nba_names[nba_comp_idx_in_filtered]
else:
    outputs = filtered_nba_vectors[candidate_indices][:, output_indices].sum(axis=1)
    floor_idx = np.argmin(outputs)
    ceiling_idx = np.argmax(outputs)
    nba_floor_player = filtered_nba_names[candidate_indices[floor_idx]]
    nba_ceiling_player = filtered_nba_names[candidate_indices[ceiling_idx]]




if len(within_tol) == 0:
    raise ValueError("No NBA players found within the height/weight tolerance of input!")

filtered_nba_vectors = nba_vectors[within_tol]
filtered_nba_names = nba_names[within_tol]
nba_comp_row = nba_df[nba_df['Player'].str.lower() == best_cbb_player.lower()]
if nba_comp_row.empty:
    raise ValueError(f"No NBA stats found for CBB comp: {best_cbb_player}")
nba_comp_vector = nba_comp_row[nba_stat_cols].values.astype(float)[0].reshape(1, -1)
comp_output = nba_comp_vector[0, output_indices].sum()

height_tol = nba_height_tol  # start with stricter
weight_tol = nba_weight_tol
found_floor = False

while not found_floor:
    # Use input player height/weight for tolerance (not NBA comp!)
    within_tol = [
        idx for idx in range(nba_vectors.shape[0])
        if abs(nba_vectors[idx, nba_stat_cols.index('Height_in')] - input_height) <= height_tol and
           abs(nba_vectors[idx, nba_stat_cols.index('Weight_lb')] - input_weight) <= weight_tol
    ]
    if len(within_tol) == 0:
        # If no candidates, relax tolerances and try again
        height_tol += 1
        weight_tol += 10
        continue

    filtered_nba_vectors = nba_vectors[within_tol]
    filtered_nba_names = nba_names[within_tol]

    # Similarity to comp
    nba_similarities = cosine_similarity(nba_comp_vector, filtered_nba_vectors)[0]
    nba_comp_idx_in_filtered = np.argmax(nba_similarities)

    # Set a similarity threshold (try 0.92 for strong style match; adjust up/down as needed)
    similarity_threshold = 0.998
    # Only keep candidates who are NOT the comp and have similarity above threshold
    candidate_indices = [
        i for i in range(len(filtered_nba_names))
        if i != nba_comp_idx_in_filtered
        and nba_similarities[i] > similarity_threshold
        and filtered_nba_names[i].strip().lower() != input_player_name_lower
    ]
    if len(candidate_indices) == 0:
        # If no candidates, relax thresholds a bit and try again
        height_tol += 1
        weight_tol += 10
        continue

    outputs = filtered_nba_vectors[candidate_indices][:, output_indices].sum(axis=1)
    floor_mask = outputs < comp_output

    if np.any(floor_mask):
        # Find lowest output among those < comp_output
        floor_candidates = np.array(candidate_indices)[floor_mask]
        floor_outputs = outputs[floor_mask]
        floor_idx = floor_candidates[np.argmin(floor_outputs)]
        nba_floor_player = filtered_nba_names[floor_idx]
        found_floor = True
    else:
        # Relax and try again
        height_tol += 1
        weight_tol += 10


# Ceiling is still max output (as before, among all candidates)
outputs = filtered_nba_vectors[candidate_indices][:, output_indices].sum(axis=1)
if len(candidate_indices) > 0:
    ceiling_idx = candidate_indices[np.argmax(outputs)]
    nba_ceiling_player = filtered_nba_names[ceiling_idx]



comp_row = cbb_df_raw[cbb_df_raw['Player'].str.lower() == best_cbb_player.lower()]

good_comp_row = cbb_df_raw[
    cbb_df_raw['Player'].str.lower() == good_cbb_player.lower()
]

display_cols = [c for c in stat_cols if c != "Explosiveness"]             
# -----------------------------------------------------------------

input_vector_trim = input_vector_full[:len(display_cols)]

# build the *input* line with just those columns
display_cols = stat_cols                       # already defined earlier

o_val = float(o_val) if o_val is not None else 100.0
d_val = float(d_val) if d_val is not None else 100.0
w_val = float(w_val) if w_val is not None else 100.0
input_ath = float(input_ath)
input_str = float(input_str)
input_quk = float(input_quk)


input_row = (
    pd.DataFrame([last_row_raw])
    .assign(
        Player      = input_player_name,
        Height_in   = float(input_height),
        Weight_lb   = float(input_weight),
        Conf_Power  = float(input_cbb_dict['Conf_Power']),
        Athleticism = float(input_ath),
        Strength    = float(input_str),
        Quickness   = float(input_quk),
        ortg        = float(o_val),
        drtg        = float(d_val),
        **{'ws/40':  float(w_val)},
    )
    .set_index('Player')
    .reindex(columns=display_cols, fill_value=0.0)
)



# the comp row already lives in cbb_df_ext – we slice the same columns:
rows = [input_row]
if not comp_row.empty:
    rows.append(comp_row.set_index('Player')[display_cols])

# add Good‑Comp row only when show_good is True
if show_good and not good_comp_row.empty \
        and good_cbb_player.lower() != best_cbb_player.lower():
    rows.append(good_comp_row.set_index('Player')[display_cols])

both_disp = pd.concat(rows, axis=0)

# Find the NBA comp vector for the best CBB comp
nba_comp_row = nba_df[nba_df['Player'].str.lower() == best_cbb_player.lower()]
if nba_comp_row.empty:
    raise ValueError(f"No NBA stats found for CBB comp: {best_cbb_player}")
nba_comp_vector = nba_comp_row[nba_stat_cols].values.astype(float)[0].reshape(1, -1)
nba_comp_height = nba_comp_row['Height_in'].values[0]
nba_comp_weight = nba_comp_row['Weight_lb'].values[0]


def format_stats_row(df, name, cols):
    row = df[df['Player'].str.lower() == name.lower()]
    if row.empty:
        return f"No stats found for {name}"
    d = row[cols].iloc[0].to_dict()
    return pd.DataFrame([d], index=[name])

input_cbb_dict = {c: v for c, v in zip(all_cols, input_vector[0])}
comp_cbb_dict = comp_row.iloc[0].to_dict()             # comp_row is the DataFrame row for the comp player

delta_cbb = np.array([
    float(input_cbb_dict[stat]) - float(comp_cbb_dict[stat])
    for stat in output_stats
])

nba_comp_stats_row = nba_comp_row[output_stats].iloc[0].values.astype(float)
nba_floor_stats_row = nba_df[nba_df['Player'].str.lower() == nba_floor_player.lower()][output_stats].iloc[0].values.astype(float)
nba_ceiling_stats_row = nba_df[nba_df['Player'].str.lower() == nba_ceiling_player.lower()][output_stats].iloc[0].values.astype(float)

good_nba_stats_row = nba_df[
    nba_df['Player'].str.lower() == good_cbb_player.lower()
][output_stats].iloc[0].values.astype(float)

stat_scaling = {
    'PTS': 0.6,
    'TRB': 0.5,
    'AST': 0.4,
    'STL': 0.4,
    'BLK': 0.4
}

scaled_delta = np.array([delta_cbb[i] * stat_scaling.get(stat, 0.5) for i, stat in enumerate(output_stats)])
predicted_nba_stats = nba_comp_stats_row + scaled_delta

pred_ppg  = float(predicted_nba_stats[output_stats.index('PTS')])

if show_good:
    good_pct = (
        dice_similarity(input_vector[0], good_vec_full)*100
        - good_cbb_distance
    )

show_good = (pred_ppg < 10) and (good_idx != best_idx) \
            and (nba_ppg_map.get(good_cbb_player, 0) >= 10)

# Clip prediction within observed floor and ceiling
nba_stat_ranges = []
for i, stat in enumerate(output_stats):
    floor    = float(nba_floor_stats_row[i])
    comp     = float(nba_comp_stats_row[i])
    good_ce  = float(good_nba_stats_row[i])
    ceiling  = float(nba_ceiling_stats_row[i])
    pred     = float(predicted_nba_stats[i])

    stat_entry = {
        "Stat"   : stat,
        "Floor"  : round(floor,   1),
        "Comp"   : round(comp,    1),
        "Pred"   : round(max(0.1, pred), 1),
        "Ceiling": round(ceiling, 1)
    }
    if show_good:
        stat_entry["GoodPred"] = round(good_ce, 1)

    nba_stat_ranges.append(stat_entry)

# --- Output everything to file ---
with open("comp_output.txt", "w") as f:

    # ---------- HEADER --------------------------------------------
    f.write("=== COMP & INPUT (CBB) ===\n")
    f.write(f"Comp: {best_cbb_player} (Similarity Percent: {dice_score_pct:.1f}%)\n")

    if show_good and good_idx != best_idx:
        good_pct = dice_similarity(input_vector[0], good_vec_full)*100 - good_cbb_distance
        f.write(f"Good Player Comp: {good_cbb_player} (Similarity Percent: {good_pct:.1f}%)\n")

    f.write(f"{input_player_name}: (input stats)\n\n")
    f.write(both_disp.to_string())

    # ---------- NBA COMPS TABLE -----------------------------------
    f.write("\n\n=== NBA COMPS (FLOOR / COMP / CEILING) ===\n")

    if show_good:
        out_df = pd.concat([
            format_stats_row(nba_df, nba_floor_player,  nba_stat_cols),
            format_stats_row(nba_df, best_cbb_player,   nba_stat_cols),
            format_stats_row(nba_df, good_cbb_player,   nba_stat_cols),
            format_stats_row(nba_df, nba_ceiling_player, nba_stat_cols),
        ])
        out_df.index = [
            f"{nba_floor_player} (Floor)",
            f"{best_cbb_player} (Comp)",
            f"{good_cbb_player} (Good Comp)",
            f"{nba_ceiling_player} (Ceiling)"
        ]
    else:
        out_df = pd.concat([
            format_stats_row(nba_df, nba_floor_player,  nba_stat_cols),
            format_stats_row(nba_df, best_cbb_player,   nba_stat_cols),
            format_stats_row(nba_df, nba_ceiling_player, nba_stat_cols),
        ])
        out_df.index = [
            f"{nba_floor_player} (Floor)",
            f"{best_cbb_player} (Comp)",
            f"{nba_ceiling_player} (Ceiling)"
        ]

    f.write(out_df.to_string())

    # ---------- PREDICTED RANGE TABLE -----------------------------
    f.write("\n\n=== PREDICTED NBA CAREER STAT RANGES ===\n")

    if show_good:
        f.write(f"{'Stat':<7}{'Floor':>8}{'Comp':>8}{'Pred':>8}{'Good Pred':>11}{'Ceiling':>10}\n")
    else:
        f.write(f"{'Stat':<7}{'Floor':>8}{'Comp':>8}{'Pred':>8}{'Ceiling':>10}\n")

    for stat_row in nba_stat_ranges:                 # ← put the loop back
        if show_good:
            f.write(
                f"{stat_row['Stat']:<7}"
                f"{stat_row['Floor']:>8}"
                f"{stat_row['Comp']:>8}"
                f"{stat_row['Pred']:>8}"
                f"{stat_row['GoodPred']:>11}"
                f"{stat_row['Ceiling']:>10}\n"
            )
        else:
            f.write(
                f"{stat_row['Stat']:<7}"
                f"{stat_row['Floor']:>8}"
                f"{stat_row['Comp']:>8}"
                f"{stat_row['Pred']:>8}"
                f"{stat_row['Ceiling']:>10}\n"
            )

print("Done! Output written to comp_output.txt")


# --- STAT ORDER DEFINITIONS ---
radar_stats = ['PTS', 'TRB', 'AST', 'BLK', 'STL']  # for CBB and NBA
nba_stats = radar_stats  # Just to be explicit

# ========== 1. INPUT VS COMP (CBB) ==========

# Input and comp player stats (CBB, in radar_stats order)
input_stats = [float(last_row_raw[stat]) for stat in radar_stats]
comp_cbb_stats = comp_row.iloc[0] if not comp_row.empty else None
comp_stats  = [float(comp_row.iloc[0][stat]) if not comp_row.empty else 0
               for stat in radar_stats]

# Find max for each CBB stat
cbb_max_stats = [cbb_df_raw[stat].max() for stat in radar_stats]
norm_input = [min(v / m, 1) if m > 0 else 0 for v, m in zip(input_stats, cbb_max_stats)]
norm_comp = [v / m if m > 0 else 0 for v, m in zip(comp_stats, cbb_max_stats)]

# Radar polygon setup
angles = np.linspace(0, 2 * np.pi, len(radar_stats) + 1)
plot_input = norm_input + [norm_input[0]]
plot_comp = norm_comp + [norm_comp[0]]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, plot_input, color='#1363DF', linewidth=2, label=input_player_name)
ax.fill(angles, plot_input, color='#47B5FF', alpha=0.3)
ax.plot(angles, plot_comp, color='#DF1313', linewidth=2, label=best_cbb_player)
ax.fill(angles, plot_comp, color='#FF4747', alpha=0.2)

grade_txt = f"Similarity Score: {dice_score_pct:.1f}/100"

anchor = AnchoredText(
    grade_txt,
    loc="lower left",        # anchor corner
    bbox_to_anchor=(-0.04, -0.06),   # ← push farther left/down
    bbox_transform=ax.transAxes,     # interpret coords in axes fractions
    pad=0.4,
    borderpad=0.5,
    prop=dict(size=10, weight='bold', color='#222'),
    frameon=True
)

# Optional styling tweaks
anchor.patch.set_edgecolor('#444')
anchor.patch.set_linewidth(0.6)
anchor.patch.set_facecolor('white')
anchor.patch.set_alpha(0.8)

ax.add_artist(anchor)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_stats, fontsize=13, fontweight='bold')
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
ax.set_ylim(0, 1)
plt.title(f"CBB: {input_player_name} vs {best_cbb_player}", size=16, pad=24, fontweight='bold')
ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05))
plt.tight_layout()
plt.savefig("input_vs_comp_radar.png", dpi=180)
plt.close()

# ========== 2. NBA FLOOR VS COMP VS CEILING VS PREDICTED ==========

def get_nba_stats(player_name):
    row = nba_df[nba_df['Player'].str.lower() == player_name.lower()]
    if not row.empty:
        return [float(row[stat].values[0]) for stat in nba_stats]
    else:
        return [0.0 for _ in nba_stats]

nba_comp_vals = get_nba_stats(best_cbb_player)
nba_floor_vals = get_nba_stats(nba_floor_player)
nba_ceiling_vals = get_nba_stats(nba_ceiling_player)
nba_max_stats = [nba_df[stat].max() for stat in nba_stats]

# Predicted NBA stats (make sure these are non-negative!)
pred_stats_dict = dict(zip(output_stats, predicted_nba_stats))
predicted_vals_ordered = [max(0.1, float(pred_stats_dict[stat])) for stat in nba_stats]

norm_comp    = [min(v / m, 1) if m > 0 else 0 for v, m in zip(comp_stats, cbb_max_stats)]
norm_floor   = [min(v / m, 1) if m > 0 else 0 for v, m in zip(nba_floor_vals, nba_max_stats)]
norm_ceiling = [min(v / m, 1) if m > 0 else 0 for v, m in zip(nba_ceiling_vals, nba_max_stats)]
norm_pred    = [min(v / m, 1) if m > 0 else 0 for v, m in zip(predicted_vals_ordered, nba_max_stats)]

plot_comp = norm_comp + [norm_comp[0]]
plot_floor = norm_floor + [norm_floor[0]]
plot_ceiling = norm_ceiling + [norm_ceiling[0]]
plot_pred = norm_pred + [norm_pred[0]]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, plot_floor, color='#2E8B57', linewidth=2, label=nba_floor_player)
ax.fill(angles, plot_floor, color='#93d8b6', alpha=0.15)
ax.plot(angles, plot_pred, color="#B42CDD", linewidth=2, label=f"{input_player_name} (Pred)")
ax.fill(angles, plot_pred, color="#DF77FF", alpha=0.18)
ax.plot(angles, plot_ceiling, color='#DF1313', linewidth=2, label=nba_ceiling_player)
ax.fill(angles, plot_ceiling, color='#FF4747', alpha=0.13)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(nba_stats, fontsize=13, fontweight='bold')
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
ax.set_ylim(0, 1)

plt.title(f"NBA: Floor vs {input_player_name} (Pred) vs Ceiling", size=15, pad=24)
ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05))
plt.tight_layout()
plt.savefig("nba_floor_comp_ceiling_radar.png", dpi=180)
plt.close()

# ========== 3. NBA COMP VS PREDICTED ==========

# Map predicted stats to correct stat order
pred_stats_dict = dict(zip(output_stats, predicted_nba_stats))
radar_stat_floors = {'PTS': 0, 'TRB': 0, 'AST': 0, 'BLK': 0.1, 'STL': 0}
predicted_vals_ordered = [
    max(radar_stat_floors.get(stat, 0), float(pred_stats_dict[stat]))
    for stat in nba_stats
]

norm_pred = [min(v / m, 1) if m > 0 else 0 for v, m in zip(predicted_vals_ordered, nba_max_stats)]
norm_comp = [min(v / m, 1) if m > 0 else 0 for v, m in zip(nba_comp_vals, nba_max_stats)]
plot_pred = norm_pred + [norm_pred[0]]
plot_comp = norm_comp + [norm_comp[0]]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, plot_comp, color='#DF1313', linewidth=2, label=f"{best_cbb_player} (Comp)")
ax.fill(angles, plot_comp, color='#FF4747', alpha=0.3)
ax.plot(angles, plot_pred, color='#1363DF', linewidth=2, label=f"{input_player_name} (Pred)")
ax.fill(angles, plot_pred, color='#47B5FF', alpha=0.2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(nba_stats, fontsize=13, fontweight='bold')
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
ax.set_ylim(0, 1)

plt.title(f"NBA: {input_player_name} (Pred) vs {best_cbb_player}", size=16, pad=24, fontweight='bold')
ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05))
plt.tight_layout()
plt.savefig("nba_comp_vs_pred_radar.png", dpi=180)
plt.close()
