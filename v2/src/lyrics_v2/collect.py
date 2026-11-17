"""Fetch lyrics for the track lists in data/tracks/*.csv.

    python -m lyrics_v2.collect --market US
    python -m lyrics_v2.collect --market GB --years 2023 2024

Primary source: LRCLIB (https://lrclib.net, public API, no token):
    GET /api/get?artist_name=X&track_name=Y  ->  ``plainLyrics``
Fallback (only for tracks LRCLIB cannot find): Genius via lyricsgenius, using the
token from GENIUS_ACCESS_TOKEN / repo-root .env. A Genius 403 (bot protection) disables
the fallback for the rest of the run; nothing tries to work around it.

Reads  data/tracks/tracks_<market>_<year>.csv  (Spotify, user supplied).
Billboard lists (tracks_BB_<year>.csv, market US) only with --include-billboard.
Writes data/lyrics/lyrics_<market>_<year>.csv  and
       data/lyrics/not_found_<market>_<year>.csv  (all git-ignored; copyrighted text).
Already-fetched / already-logged tracks are skipped, so the command is resumable.
"""
import argparse
import csv
import os
import re
import sys
import time

from .paths import LYRICS_DIR, REPO_ROOT, TRACKS_DIR

LYRIC_COLUMNS = ["year", "market", "source", "artist", "track", "popularity",
                 "genre", "lyrics_source", "lyrics"]
NOT_FOUND_COLUMNS = ["year", "market", "source", "artist", "track", "reason"]
LRCLIB_URL = "https://lrclib.net/api/get"
USER_AGENT = "lyrics-topic-modeling/2.0 (research project; lyrics_v2.collect)"
REQUEST_DELAY = 0.5
_SUFFIX = re.compile(r"\s*[\(\[-].*(remix|version|live|feat|ft\.|edit|remaster).*$", re.I)


def clean_title(title):
    """Strip remix/version/feat suffixes so the search matches the base song."""
    return _SUFFIX.sub("", title).strip()


def discover_track_files(market, years=None, include_billboard=False):
    """Return [(path, source, year)] of track lists for a market."""
    files = []
    if not TRACKS_DIR.exists():
        return files
    patterns = [(f"tracks_{market}_", "spotify")]
    if include_billboard and market.upper() == "US":
        patterns.append(("tracks_BB_", "billboard"))
    for path in sorted(TRACKS_DIR.glob("tracks_*.csv")):
        for prefix, source in patterns:
            m = re.fullmatch(re.escape(prefix) + r"(\d{4})\.csv", path.name)
            if m and (not years or int(m.group(1)) in years):
                files.append((path, source, int(m.group(1))))
    return files


def read_tracks(path):
    """Read a track list. If popularity carries no signal (all values equal, e.g.
    the Spotify API returned none), fall back to file order (search rank, first = best)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len({r.get("popularity", "") for r in rows}) <= 1:
        n = len(rows)
        for i, r in enumerate(rows):
            r["popularity"] = n - i
    return rows


def load_done(path):
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["artist"].lower(), r["track"].lower()) for r in csv.DictReader(fh)}


def get_token(env_file=None):
    """Genius token from the environment, else from the git-ignored repo-root .env.

    Accepted names: GENIUS_ACCESS_TOKEN (preferred) or GENIUS_ACCESS. The value is
    only held in memory; it is never printed or written anywhere.
    """
    names = ("GENIUS_ACCESS_TOKEN", "GENIUS_ACCESS")
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    env_file = env_file or (REPO_ROOT / ".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip().removeprefix("export ").strip().upper() in names:
                return val.strip().strip("'\"")
    return None


def fetch_lrclib(session, artist, track, retries=3):
    """Return (plain_lyrics or None, reason). reason is '' on success."""
    last = "error"
    for attempt in range(retries):
        try:
            r = session.get(LRCLIB_URL, params={"artist_name": artist, "track_name": track},
                            timeout=20)
        except Exception as exc:  # noqa: BLE001 - network hiccup, retry
            last = f"network: {type(exc).__name__}"
            time.sleep(REQUEST_DELAY * (attempt + 1) * 2)
            continue
        if r.status_code == 404:
            return None, "lrclib 404"
        if r.status_code == 200:
            data = r.json()
            if data.get("instrumental"):
                return None, "instrumental"
            text = data.get("plainLyrics")
            return (text, "") if text else (None, "no plainLyrics")
        last = f"http {r.status_code}"
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(REQUEST_DELAY * (attempt + 1) * 4)
            continue
        break
    return None, last


class GeniusFallback:
    """Lazy Genius client; disables itself on 403 or when no token is available."""

    def __init__(self, token):
        self.token, self.client, self.disabled = token, None, token is None

    def fetch(self, artist, track):
        if self.disabled:
            return None
        try:
            if self.client is None:
                import lyricsgenius

                self.client = lyricsgenius.Genius(
                    self.token, timeout=15, retries=1, sleep_time=REQUEST_DELAY,
                    remove_section_headers=True, verbose=False)
            song = self.client.search_song(clean_title(track), artist)
            return song.lyrics if song and song.lyrics else None
        except Exception as exc:  # noqa: BLE001
            if "403" in str(exc):
                self.disabled = True
                print("  Genius fallback disabled for this run (403 / bot protection).")
            return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch lyrics for listed tracks (LRCLIB, Genius fallback)")
    ap.add_argument("--market", required=True, help="US, GB, ...")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025],
                    help="Target years (default 2023 2024 2025)")
    ap.add_argument("--include-billboard", action="store_true",
                    help="Also read tracks_BB_<year>.csv (US)")
    ap.add_argument("--no-genius", action="store_true", help="Skip the Genius fallback")
    ap.add_argument("--limit", type=int, help="Max tracks per file (smoke tests)")
    args = ap.parse_args(argv)

    market = args.market.upper()
    files = discover_track_files(market, args.years, args.include_billboard)
    if not files:
        print(f"No track lists for market {args.market} under {TRACKS_DIR}. "
              "Add Spotify CSVs (tracks_<market>_<year>.csv) first.", file=sys.stderr)
        return 2

    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    genius = GeniusFallback(None if args.no_genius else get_token())
    LYRICS_DIR.mkdir(parents=True, exist_ok=True)

    for path, source, year in files:
        out = LYRICS_DIR / f"lyrics_{market}_{year}.csv"
        nf_out = LYRICS_DIR / f"not_found_{market}_{year}.csv"
        done = load_done(out) | load_done(nf_out)
        counts = {"lrclib": 0, "genius": 0, "missing": 0}
        tracks = read_tracks(path)[: args.limit]
        with open(out, "a", newline="", encoding="utf-8") as fh, \
                open(nf_out, "a", newline="", encoding="utf-8") as nf:
            w, nfw = csv.writer(fh), csv.writer(nf)
            if out.stat().st_size == 0:
                w.writerow(LYRIC_COLUMNS)
            if nf_out.stat().st_size == 0:
                nfw.writerow(NOT_FOUND_COLUMNS)
            for t in tracks:
                if (t["artist"].lower(), t["track"].lower()) in done:
                    continue
                text, reason = fetch_lrclib(session, t["artist"], t["track"])
                time.sleep(REQUEST_DELAY)
                lsource = "lrclib"
                cleaned = clean_title(t["track"])
                if not text and cleaned and cleaned != t["track"]:
                    text, reason = fetch_lrclib(session, t["artist"], cleaned)
                    time.sleep(REQUEST_DELAY)
                if not text and reason != "instrumental":
                    text = genius.fetch(t["artist"], t["track"])
                    lsource = "genius"
                    if text:
                        reason = ""
                if text:
                    counts[lsource] += 1
                    w.writerow([year, market, source, t["artist"], t["track"],
                                t.get("popularity", ""), "", lsource, text])
                    fh.flush()
                else:
                    counts["missing"] += 1
                    nfw.writerow([year, market, source, t["artist"], t["track"], reason])
                    nf.flush()
        print(f"[{year}/{source}] lrclib {counts['lrclib']}, genius {counts['genius']}, "
              f"not found {counts['missing']} -> {out.name}, {nf_out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
