# Shared “magic numbers” so every module agrees
DERIVED_COLS = [
    "AST_to_TOV",
    "3PA_rate",
    "Stocks40",
    "TS",
    "USG40",
    "CreationLoad",
    "AST_per_FGA",
    "FT_rate",
    "SelfCreationIdx",
]

COUNTING_STATS = [
    "PTS", "TRB", "AST", "STL", "BLK", "ORB", "DRB",
    "FGA", "FGM", "3P", "3PA", "FTA", "FT", "TOV", "PF", "2P", "2PA",
]

POWER_CONFS = {
    "ACC", "SEC", "BIG TEN", "BIG 12", "BIG EAST", "PAC-12", "PAC-10",
}

DEFAULT_METRIC = 100.0
MISSING_FIELD_VALUE = 6
