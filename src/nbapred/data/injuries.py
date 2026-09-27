"""Official NBA injury reports: fetch, parse, and pick the report that was public before tip-off.

Reports live at ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<slot>.pdf. The slot format
changed over time: '05PM' (published ~5:30 PM ET) up to 2025, then '05_30PM' every 15 minutes. Each fetched
report is parsed to rows (player, team, status) and cached, including misses, so nothing is fetched twice.
"""
import io
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta

import pandas as pd
import pdfplumber
import requests
from nba_api.stats.static import teams as static_teams

from nbapred.config import RAW

URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{day}_{slot}.pdf"
CACHE = RAW / "injury"
STATUSES = ["Out", "Doubtful", "Questionable", "Probable", "Available"]
LEAD = timedelta(minutes=60)   # a report must be public this long before tip-off to be used
_GAP_S = 0.25
_last = 0.0

# "Last, First" (old layout, spaced) or "Last,First" (new layout, squished); last names may have spaces/suffixes
_NAME_RE = re.compile(r"(?P<name>[A-Z][A-Za-z.'\-]*(?: [A-Z][A-Za-z.'\-]*)*, ?[A-Z][A-Za-z.'\-]+)")
_STATUS_WORD = re.compile(r"\b(Out|Doubtful|Questionable|Probable|Available)\b")
_STAMP_RE = re.compile(r"Injury Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{1,2}:\d{2})\s*(AM|PM)")
_TEAMS = {t["full_name"].replace(" ", ""): t["id"] for t in static_teams.get_teams()}


NEW_FORMAT_FROM = datetime(2025, 10, 1)  # quarter-hourly '05_30PM' files; before that hourly '05PM' files


def _slots_before(deadline: datetime) -> list[tuple[str, datetime]]:
    """Candidate report slots (label, publication time) at or before `deadline`, latest first.
    Hourly files ('05PM') are stamped at :30. From 2025-26 files are quarter-hourly ('05_30PM')."""
    out = []
    day = deadline.replace(hour=0, minute=0, second=0, microsecond=0)
    new = deadline >= NEW_FORMAT_FROM
    for minutes in range(20 * 60 + 30, 10 * 60, -15):  # 8:30 PM back to 10:15 AM
        t = day + timedelta(minutes=minutes)
        if t > deadline:
            continue
        h12, ampm = (t.hour - 1) % 12 + 1, ("AM" if t.hour < 12 else "PM")
        if new:
            out.append((f"{h12:02d}_{t.minute:02d}{ampm}", t))
        if t.minute == 30:
            out.append((f"{h12:02d}{ampm}", t))
    return out


def _fetch(day: str, slot: str) -> pd.DataFrame | None:
    """Parsed rows for one report, or None if it doesn't exist. Cached either way."""
    global _last
    path = CACHE / f"{day}_{slot}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return None if df.empty and "missing" in df.columns else df
    wait = _GAP_S - (time.time() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.time()
    r = requests.get(URL.format(day=day, slot=slot), timeout=30)
    CACHE.mkdir(parents=True, exist_ok=True)
    if r.status_code != 200:
        _write(pd.DataFrame({"missing": pd.Series(dtype=bool)}), path)
        return None
    df = parse(r.content)
    _write(df, path)
    return df


def _write(df: pd.DataFrame, path) -> None:
    """Atomic write, so parallel backfill workers never leave a half-written file."""
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def parse(pdf_bytes: bytes) -> pd.DataFrame:
    """PDF -> rows (report_time, team_id, raw_name, name_key, status).

    Handles both layouts:
      2019-2024ish: 'Covington, Robert Injury/Illness Right Knee Bone Bruise Out -'  (current, previous status)
      later:        'Brunson,Jalen Probable Injury/Illness-RightShoulder;'            (status right after name)
    Reasons can wrap, so a name whose status lands on the next line is held as pending.
    team_id is best-effort (team names also wrap); callers should resolve teams by player name.
    """
    rows, team_id, stamp, pending, old_layout = [], None, None, None, False
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").splitlines():
                if stamp is None and (m := _STAMP_RE.search(line)):
                    stamp = datetime.strptime(" ".join(m.groups()), "%m/%d/%y %I:%M %p")
                    continue
                if "Previous Status" in line:
                    old_layout = True
                    continue
                squished = line.replace(" ", "")
                for name, tid in _TEAMS.items():
                    if name in squished:
                        team_id = tid
                name_m = _NAME_RE.search(line)
                statuses = _STATUS_WORD.findall(line)
                if old_layout and statuses:
                    # trailing columns are "<current> <previous|->"; current is the last status before them
                    tail = line.split()[-2:]
                    status = tail[0] if tail[0] in STATUSES else statuses[-1]
                else:
                    status = statuses[0] if statuses else None
                if name_m and status:
                    rows.append((team_id, _clean_name(name_m["name"]), status))
                    pending = None
                elif name_m:
                    pending = (team_id, _clean_name(name_m["name"]))
                elif status and pending:
                    rows.append((*pending, status))
                    pending = None
    df = pd.DataFrame(rows, columns=["team_id", "raw_name", "status"])
    df["name_key"] = df["raw_name"].map(name_key)
    return df.assign(report_time=stamp)


_SUFFIXES = {"Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV"}
_PARTICLES = {"Van", "Von", "Da", "De", "Del", "Di", "Du", "La", "Le", "St.", "Mac"}


def _clean_name(raw: str) -> str:
    """Drop matchup/team words glued in front of a name: 'PHI Minnesota Covington, Robert' -> 'Covington, Robert'."""
    last, first = raw.split(",", 1)
    words = last.split()
    keep = [words.pop()]
    if keep[0] in _SUFFIXES and words:
        keep.insert(0, words.pop())
    while words and words[-1] in _PARTICLES:
        keep.insert(0, words.pop())
    return " ".join(keep) + "," + first


def name_key(name: str) -> str:
    """'McCullarJr.,Kevin' / 'Kevin McCullar Jr.' / 'Dončić, Luka' -> comparable key."""
    if "," in name:
        last, first = name.split(",", 1)
        name = first + last
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"(Jr|Sr|II|III|IV)\.?$", "", re.sub(r"[^A-Za-z]", "", name.replace("Jr.", "Jr")))
    return name.lower()


def report_before(tipoff: datetime, offline: bool = False) -> pd.DataFrame | None:
    """The latest report published at least LEAD before `tipoff` (ET, naive). None if none found that day.
    offline=True only reads the local cache (for model building); a slot never fetched counts as missing."""
    deadline = tipoff - LEAD
    day = deadline.strftime("%Y-%m-%d")
    for slot, _ in _slots_before(deadline):
        if offline and not (CACHE / f"{day}_{slot}.parquet").exists():
            continue
        df = _fetch(day, slot)
        if df is not None:
            return df.assign(slot=slot)
    return None


ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_STATUS = {"Out": "Out", "Doubtful": "Doubtful", "Questionable": "Questionable", "Day-To-Day": "Questionable",
               "Probable": "Probable"}


def latest_report(now: datetime) -> pd.DataFrame | None:
    """Most recent official report published at or before `now` (ET) today, fetched live."""
    day = now.strftime("%Y-%m-%d")
    for slot, _ in _slots_before(now):
        df = _fetch(day, slot)
        if df is not None:
            return df.assign(slot=slot)
    return None


def espn_report() -> pd.DataFrame:
    """Fallback: ESPN's current injury list, in the same shape as a parsed official report."""
    data = requests.get(ESPN_URL, timeout=30).json()
    rows = []
    for team in data.get("injuries", []):
        for inj in team.get("injuries", []):
            status = ESPN_STATUS.get(inj.get("status"))
            name = inj.get("athlete", {}).get("displayName", "")
            if status and name:
                tid = _TEAMS.get(team.get("displayName", "").replace(" ", ""))
                rows.append({"team_id": tid, "raw_name": name, "name_key": name_key(name), "status": status})
    return pd.DataFrame(rows, columns=["team_id", "raw_name", "name_key", "status"]).assign(
        report_time=datetime.now(), slot="espn")
