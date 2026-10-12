#!/usr/bin/env python3
"""
NBA injury alerts -> Telegram, with teammate impact computed from YOUR game logs.

What it does
  1. Polls ESPN's public injury feed every N seconds (unofficial feed: can lag or break).
  2. Detects new entries, status changes, changed notes, and players removed from the report.
  3. For each change, looks up the player in nba_gamelogs.csv (made by the model project)
     and computes how his teammates performed WITH vs WITHOUT him (minutes and PRA).
  4. Sends one Telegram message per change. Without Telegram credentials it prints instead.

Setup (once)
  pip install pandas numpy requests
  1) In Telegram, open @BotFather -> /newbot -> copy the TOKEN.
  2) Open your new bot and send it any message (for example: hi).
  3) export TELEGRAM_TOKEN="123:ABC..."
     python nba_alerts.py --find-chat-id        # prints your chat id
     export TELEGRAM_CHAT_ID="123456789"
  4) python nba_alerts.py --selftest            # offline check that everything works
  5) python nba_alerts.py                       # run the monitor (keep it running)

On Colab use os.environ["TELEGRAM_TOKEN"]="..." in a cell before running the script
with !python. Colab disconnects after a while, so it is only good for testing.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
import unicodedata

import numpy as np
import pandas as pd
import requests

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
SNAPSHOT = "injury_snapshot.json"
UA = {"User-Agent": "Mozilla/5.0"}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`,\-]", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def send(text, token, chat_id):
    if not token or not chat_id:
        print("[console]\n" + text + "\n")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": text}, timeout=15)
        if not r.ok:
            print("Telegram error", r.status_code, r.text[:200])
    except Exception as e:  # network hiccup: do not crash the monitor
        print("Telegram send failed:", e)


def find_chat_id(token):
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=15)
    ids = set()
    for u in r.json().get("result", []):
        msg = u.get("message") or u.get("channel_post") or {}
        chat = msg.get("chat") or {}
        if "id" in chat:
            ids.add(chat["id"])
    print("Chat ids found:", ids if ids else "none (send your bot a message first)")


# ----------------------------------------------------------------------------
# injury feed
# ----------------------------------------------------------------------------
def fetch_injuries():
    r = requests.get(ESPN_URL, headers=UA, timeout=15)
    r.raise_for_status()
    out = {}
    for group in r.json().get("injuries", []):
        for inj in group.get("injuries", []):
            a = inj.get("athlete") or {}
            name = a.get("displayName")
            if not name:
                continue
            team = (a.get("team") or {}).get("abbreviation") or group.get("displayName", "")
            d = inj.get("details") or {}
            out[name] = dict(team=team,
                             status=inj.get("status", ""),
                             comment=inj.get("shortComment", ""),
                             date=inj.get("date", ""),
                             body=d.get("type", ""),
                             ret=d.get("returnDate", ""))
    return out


def diff(old, new):
    """Returns list of (name, info, old_info, kind)."""
    ev = []
    for n, v in new.items():
        o = old.get(n)
        if o is None:
            ev.append((n, v, None, "NEW"))
        elif o["status"] != v["status"]:
            ev.append((n, v, o, "STATUS"))
        elif o["comment"] != v["comment"]:
            ev.append((n, v, o, "NOTE"))
    for n, o in old.items():
        if n not in new:
            ev.append((n, o, o, "REMOVED"))
    return ev


# ----------------------------------------------------------------------------
# teammate impact from your own game logs
# ----------------------------------------------------------------------------
class Impact:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path, parse_dates=["GAME_DATE"])
        df = df[df["MIN"] > 0].copy()
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        df["key"] = df["PLAYER_NAME"].map(norm)
        self.df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    def recent_minutes(self, key, n=15):
        rows = self.df[self.df["key"] == key].tail(n)
        return float(rows["MIN"].mean()) if len(rows) else None

    def teammates(self, key, top=3, min_without=5):
        d = self.df
        prow = d[d["key"] == key]
        if prow.empty:
            return None
        team = prow.iloc[-1]["TEAM_ABBREVIATION"]
        p_team = prow[prow["TEAM_ABBREVIATION"] == team]
        td = d[(d["TEAM_ABBREVIATION"] == team) & (d["GAME_DATE"] >= p_team["GAME_DATE"].min())]
        with_games = set(p_team["GAME_ID"])
        wo_games = set(td["GAME_ID"]) - with_games
        res = dict(team=team, n_without=len(wo_games), rows=[])
        if len(wo_games) < min_without:
            return res
        for _, g in td[td["key"] != key].groupby("key"):
            w = g[g["GAME_ID"].isin(with_games)]
            wo = g[g["GAME_ID"].isin(wo_games)]
            if len(w) < 5 or len(wo) < 4:
                continue
            res["rows"].append(dict(name=g["PLAYER_NAME"].iloc[-1], n=len(wo),
                                    min_w=w["MIN"].mean(), min_wo=wo["MIN"].mean(),
                                    pra_w=w["PRA"].mean(), pra_wo=wo["PRA"].mean()))
        res["rows"].sort(key=lambda r: r["pra_wo"] - r["pra_w"], reverse=True)
        res["rows"] = res["rows"][:top]
        return res


# ----------------------------------------------------------------------------
# message
# ----------------------------------------------------------------------------
KIND_AR = {"NEW": "إصابة جديدة", "STATUS": "تغيّر الحالة", "NOTE": "تحديث ملاحظة",
           "REMOVED": "أُزيل من التقرير (غالبًا متاح)"}


def build_message(name, info, old, kind, impact):
    lines = [f"🚨 {KIND_AR[kind]}", f"{name} ({info['team']})"]
    if kind == "STATUS" and old:
        lines.append(f"الحالة: {old['status']} ← {info['status']}")
    elif kind != "REMOVED":
        lines.append(f"الحالة: {info['status']}")
    if info.get("body"):
        lines.append(f"الإصابة: {info['body']}" + (f" | عودة متوقعة: {info['ret']}" if info.get("ret") else ""))
    if info.get("comment"):
        lines.append(info["comment"][:300])
    if info.get("date"):
        lines.append(f"وقت الخبر (ESPN): {info['date']}")

    if impact is not None:
        t = impact.teammates(norm(name))
        if t and t["rows"]:
            lines.append("")
            lines.append(f"أثر غيابه على زملائه (من {t['n_without']} مباراة غاب فيها):")
            for r in t["rows"]:
                lines.append(f"• {r['name']}: دقائق {r['min_w']:.1f}→{r['min_wo']:.1f} | "
                             f"PRA {r['pra_w']:.1f}→{r['pra_wo']:.1f} (n={r['n']})")
            lines.append("عينات صغيرة غالبًا. قارن الخط الحي بعتبتك قبل أي قرار.")
        elif t:
            lines.append("")
            lines.append(f"لا توجد عينة كافية لغيابه في بياناتك ({t['n_without']} مباراة).")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------
def load_snapshot():
    if os.path.exists(SNAPSHOT):
        with open(SNAPSHOT) as f:
            return json.load(f)
    return None


def save_snapshot(s):
    with open(SNAPSHOT, "w") as f:
        json.dump(s, f)


def handle(events, impact, args, token, chat):
    teams = {t.upper() for t in args.teams} if args.teams else None
    for name, info, old, kind in events:
        if teams and info["team"].upper() not in teams:
            continue
        mins = impact.recent_minutes(norm(name)) if impact is not None else None
        if impact is not None:
            if mins is None and not args.include_unknown:
                continue
            if mins is not None and mins < args.min_minutes:
                continue
        send(build_message(name, info, old, kind, impact), token, chat)


def run(args):
    token, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    impact = Impact(args.csv) if os.path.exists(args.csv) else None
    if impact is None:
        print(f"Note: {args.csv} not found; alerts will have no teammate-impact section.")
    old = load_snapshot()
    backoff = 0
    while True:
        try:
            new = fetch_injuries()
            if not new and old:
                print("Empty feed, skipping this poll")
            elif old is None:
                print(f"Baseline saved ({len(new)} players). Alerts start on the next change.")
                if args.alert_existing:
                    handle(diff({}, new), impact, args, token, chat)
                save_snapshot(new)
                old = new
            else:
                events = diff(old, new)
                if events:
                    handle(events, impact, args, token, chat)
                    save_snapshot(new)
                    old = new
            backoff = 0
        except Exception as e:
            backoff = min(300, max(30, backoff * 2))
            print(f"Poll failed ({e}); retrying in {backoff}s")
            time.sleep(backoff)
            continue
        if args.once:
            return
        time.sleep(args.interval)


# ----------------------------------------------------------------------------
# offline self-test
# ----------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(1)
    rows, gid = [], 0
    for g in range(60):
        gid += 1
        star_plays = not (20 <= g < 35)           # star misses 15 games mid-sample
        date = pd.Timestamp("2025-11-01") + pd.Timedelta(days=g)
        cast = [("Star Player", 36, 30)] if star_plays else []
        cast += [("Second Guy", 28 if star_plays else 34, 18 if star_plays else 24),
                 ("Bench Guy", 18 if star_plays else 22, 8 if star_plays else 11)]
        for nm, m, p in cast:
            mins = float(np.clip(rng.normal(m, 2), 5, 42))
            rows.append(dict(GAME_ID=gid, GAME_DATE=date, PLAYER_ID=hash(nm) % 9999,
                             PLAYER_NAME=nm, TEAM_ABBREVIATION="AAA", MATCHUP="AAA vs. BBB",
                             MIN=mins, PTS=int(rng.normal(p, 3)), REB=int(rng.normal(5, 1)),
                             AST=int(rng.normal(4, 1))))
    path = os.path.join(tempfile.gettempdir(), "selftest_logs.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    imp = Impact(path)

    old = {"Star Player": dict(team="AAA", status="Questionable", comment="Ankle soreness",
                               date="2026-10-12T18:00Z", body="Ankle", ret="")}
    new = {"Star Player": dict(team="AAA", status="Out", comment="Ruled out tonight",
                               date="2026-10-12T20:00Z", body="Ankle", ret="2026-10-14"),
           "Rookie X": dict(team="AAA", status="Day-To-Day", comment="", date="", body="", ret="")}
    ev = diff(old, new)
    kinds = sorted(k for *_, k in ev)
    assert kinds == ["NEW", "STATUS"], kinds
    assert [k for *_, k in diff(new, {})].count("REMOVED") == 2
    print("diff OK:", kinds)
    ns = argparse.Namespace(teams=None, min_minutes=18, include_unknown=False)
    handle(ev, imp, ns, None, None)   # Rookie X is skipped (unknown), Star Player prints
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nba_gamelogs.csv", help="game logs from the model project")
    ap.add_argument("--interval", type=int, default=60, help="seconds between polls")
    ap.add_argument("--min-minutes", type=float, default=18,
                    help="ignore players averaging fewer recent minutes")
    ap.add_argument("--teams", nargs="*", help="only these ESPN team codes, e.g. ATL BOS")
    ap.add_argument("--include-unknown", action="store_true",
                    help="also alert for players missing from your CSV")
    ap.add_argument("--alert-existing", action="store_true",
                    help="on the first run, alert for everything currently listed")
    ap.add_argument("--once", action="store_true", help="poll one time and exit")
    ap.add_argument("--find-chat-id", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.find_chat_id:
        tok = os.environ.get("TELEGRAM_TOKEN")
        return find_chat_id(tok) if tok else sys.exit("Set TELEGRAM_TOKEN first")
    run(args)


if __name__ == "__main__":
    main()
