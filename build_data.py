"""Builds nba_gamelogs.csv from the public llimllib/nba_data parquet files."""
import re
import sys
import pandas as pd

BASE = "https://raw.githubusercontent.com/llimllib/nba_data/main/data/"
SEASONS = [2025, 2026]


def pick(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def parse_min(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    m = re.match(r"PT(?:(\d+)M)?(?:([\d.]+)S)?", s)
    if m and s.startswith("PT"):
        return float(m.group(1) or 0) + float(m.group(2) or 0) / 60
    if ":" in s:
        a, b = s.split(":")[:2]
        return float(a) + float(b) / 60
    try:
        return float(s)
    except ValueError:
        return 0.0


def main():
    pl = pd.concat([pd.read_parquet(f"{BASE}playerlog_{y}.parquet") for y in SEASONS], ignore_index=True)
    gl = pd.concat([pd.read_parquet(f"{BASE}gamelog_{y}.parquet") for y in SEASONS], ignore_index=True)
    print("playerlog columns:", list(pl.columns))
    print("gamelog columns:", list(gl.columns))

    c = dict(gid=pick(pl, "GAME_ID", "gameId"), pid=pick(pl, "PLAYER_ID", "personId", "playerId"),
             name=pick(pl, "PLAYER_NAME", "playerName", "name"),
             team=pick(pl, "TEAM_ABBREVIATION", "teamTricode", "team"),
             mins=pick(pl, "MIN", "minutes"), pts=pick(pl, "PTS", "points"),
             reb=pick(pl, "REB", "reboundsTotal", "rebounds"), ast=pick(pl, "AST", "assists"),
             date=pick(pl, "GAME_DATE", "gameDate"), match=pick(pl, "MATCHUP", "matchup"))
    if c["date"] is None or c["match"] is None:
        g_id = pick(gl, "GAME_ID", "gameId")
        g_date = pick(gl, "GAME_DATE", "gameDate")
        g_team = pick(gl, "TEAM_ABBREVIATION", "teamTricode")
        g_match = pick(gl, "MATCHUP", "matchup")
        cols = [x for x in (g_id, g_team, g_date, g_match) if x]
        gi = gl[cols].drop_duplicates([x for x in (g_id, g_team) if x])
        keys = [(c["gid"], g_id)] + ([(c["team"], g_team)] if c["team"] and g_team else [])
        pl = pl.merge(gi, left_on=[a for a, _ in keys], right_on=[b for _, b in keys],
                      how="left", suffixes=("", "_g"))
        c["date"] = c["date"] or g_date
        c["match"] = c["match"] or g_match
        c["team"] = c["team"] or g_team
    missing = [k for k, v in c.items() if v is None or v not in pl.columns]
    if missing:
        sys.exit(f"Could not find columns for {missing}. See the column lists above.")

    pl = pl[pl[c["gid"]].astype(str).str.startswith("002")]
    out = pd.DataFrame({
        "GAME_ID": pl[c["gid"]], "GAME_DATE": pd.to_datetime(pl[c["date"]]).dt.strftime("%Y-%m-%d"),
        "PLAYER_ID": pl[c["pid"]], "PLAYER_NAME": pl[c["name"]],
        "TEAM_ABBREVIATION": pl[c["team"]], "MATCHUP": pl[c["match"]],
        "MIN": pl[c["mins"]].map(parse_min), "PTS": pl[c["pts"]], "REB": pl[c["reb"]], "AST": pl[c["ast"]],
    }).dropna(subset=["PLAYER_NAME"])
    out.to_csv("nba_gamelogs.csv", index=False)
    print(f"DONE rows={len(out)} players={out['PLAYER_NAME'].nunique()}")


if __name__ == "__main__":
    main()
