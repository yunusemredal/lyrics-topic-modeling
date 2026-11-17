"""Fetch Billboard Hot 100 year-end lists (no API key) into the track CSV contract.

    python -m lyrics_v2.charts --years 2023 2024 2025 2026

Output: data/tracks/tracks_BB_<year>.csv with columns
    year, market, artist, track, album, popularity, release_date, spotify_id
``popularity`` is the inverse chart rank (rank 1 -> highest value); ``album``,
``release_date`` and ``spotify_id`` stay empty.

Year-end charts are used when available; for a year that is not finished (or
has no year-end chart yet) the weekly Hot 100 charts are aggregated instead
(best peak position per song).
"""
import argparse
import csv
import sys
from datetime import date, timedelta

from .paths import TRACKS_DIR, TRACK_COLUMNS


def rows_from_entries(entries, year, market="US"):
    """Convert ordered (artist, title) pairs (best first) to contract rows."""
    seen, rows = set(), []
    for artist, title in entries:
        key = (artist.strip().lower(), title.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append([year, market, artist.strip(), title.strip(), "", 0, "", ""])
    n = len(rows)
    for i, r in enumerate(rows):
        r[5] = n - i  # inverse rank
    return rows


def fetch_year(year):
    import billboard

    try:
        chart = billboard.ChartData("hot-100-songs", year=str(year))
        entries = [(e.artist, e.title) for e in chart]
        if entries:
            return entries, "year-end"
    except Exception as exc:  # noqa: BLE001 - fall back to weekly aggregation
        print(f"[{year}] year-end chart unavailable ({exc}); aggregating weekly charts")

    best, ok, failed = {}, 0, 0
    day, end = date(year, 1, 1), min(date(year, 12, 31), date.today())
    while day <= end:
        try:
            chart = billboard.ChartData("hot-100", date=day.isoformat())
            for e in chart:
                k = (e.artist, e.title)
                best[k] = min(best.get(k, 999), e.rank)
            ok += 1
        except Exception as exc:  # noqa: BLE001 - skip unparsable weeks
            failed += 1
            print(f"[{year}] week of {day} skipped: {exc}")
        day += timedelta(days=7)
    print(f"[{year}] weekly charts parsed: {ok}, skipped: {failed}")
    return sorted(best, key=lambda k: best[k]), "weekly-aggregate"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch Billboard Hot 100 lists")
    ap.add_argument("--years", type=int, nargs="+", required=True)
    args = ap.parse_args(argv)

    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        entries, mode = fetch_year(year)
        rows = rows_from_entries(entries, year)
        if not rows:
            print(f"[{year}] no tracks retrieved; nothing written", file=sys.stderr)
            continue
        out = TRACKS_DIR / f"tracks_BB_{year}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(TRACK_COLUMNS)
            w.writerows(rows)
        print(f"[{year}] {len(rows)} tracks ({mode}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

