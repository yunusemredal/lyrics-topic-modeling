# -*- coding: utf-8 -*-
"""Collect top tracks per year/market from the Spotify search endpoint.

Usage (credentials come from environment variables, never from code):

    set SPOTIPY_CLIENT_ID=...          (PowerShell: $env:SPOTIPY_CLIENT_ID="...")
    set SPOTIPY_CLIENT_SECRET=...
    python scripts/collect_tracks.py --year 2023 --market US
    python scripts/collect_tracks.py --year 2023 --market GB --limit 400

Output: data/tracks/tracks_<market>_<year>.csv with columns
    year, market, artist, track, album, popularity, release_date, spotify_id

Results are sorted by Spotify popularity (0-100) descending and de-duplicated
by (artist, track). Note: `search` returns relevance-ranked results, not an
official chart; sorting by popularity approximates "most listened".
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except ImportError:
    sys.exit("spotipy is not installed. Run: pip install spotipy")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch top tracks for a year/market via Spotify search")
    ap.add_argument("--year", type=int, required=True, help="Release year, e.g. 2023")
    ap.add_argument("--market", default="US", help="ISO market code: US, GB, TR ... (default US)")
    ap.add_argument("--limit", type=int, default=300, help="Max tracks to keep (default 300, cap 1000)")
    ap.add_argument("--out-dir", default="data/tracks", help="Output directory (default data/tracks)")
    args = ap.parse_args()

    cid = os.environ.get("SPOTIPY_CLIENT_ID")
    secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    if not cid or not secret:
        sys.exit("Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET environment variables first.")

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(client_id=cid, client_secret=secret),
        retries=3,
        requests_timeout=15,
    )

    target = min(args.limit, 1000)  # search offset hard cap
    query = f"year:{args.year}"

    # Newer Spotify apps reject larger page sizes with "Invalid limit";
    # probe downwards until one works.
    page = None
    for candidate in (50, 20, 10, 5, 1):
        try:
            sp.search(q=query, type="track", market=args.market, limit=candidate, offset=0)
            page = candidate
            break
        except spotipy.SpotifyException as e:
            if e.http_status == 400:
                print(f"limit={candidate} rejected, trying smaller...", file=sys.stderr)
                continue
            sys.exit(f"Spotify error while probing page size: {e}")
    if page is None:
        sys.exit("Spotify rejected every page size - the app may lack search access.")
    print(f"Using page size {page}")

    seen = {}
    offset = 0
    while offset < target:
        try:
            res = sp.search(q=query, type="track", market=args.market, limit=page, offset=offset)
        except spotipy.SpotifyException as e:
            print(f"Spotify error at offset {offset}: {e}", file=sys.stderr)
            break
        items = res.get("tracks", {}).get("items", [])
        if not items:
            break
        offset += page
        for t in items:
            artist = ", ".join(a["name"] for a in t.get("artists", []))
            key = (artist.lower(), t["name"].lower())
            if key in seen:
                continue
            seen[key] = {
                "year": args.year,
                "market": args.market,
                "artist": artist,
                "track": t["name"],
                "album": (t.get("album") or {}).get("name", ""),
                "popularity": t.get("popularity", 0),
                "release_date": (t.get("album") or {}).get("release_date", ""),
                "spotify_id": t.get("id", ""),
            }
        print(f"  fetched {offset} (unique so far: {len(seen)})")
        time.sleep(0.2)  # be polite to the rate limiter

    if not seen:
        sys.exit("No tracks returned - check credentials, market code and year.")

    rows = sorted(seen.values(), key=lambda r: r["popularity"], reverse=True)[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tracks_{args.market}_{args.year}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} tracks -> {out_path}")
    print(f"Top 5 by popularity:")
    for r in rows[:5]:
        print(f"  {r['popularity']:3d}  {r['artist']} - {r['track']}")


if __name__ == "__main__":
    main()
