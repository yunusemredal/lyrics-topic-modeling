from lyrics_v2 import charts, collect


class FakeResp:
    def __init__(self, status, payload=None):
        self.status_code, self._p = status, payload or {}

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


def test_rows_from_entries_dedup_and_inverse_rank():
    rows = charts.rows_from_entries([("A", "X"), ("B", "Y"), ("a", "x")], 2023)
    assert len(rows) == 2
    assert rows[0][5] == 2 and rows[1][5] == 1


def test_empty_entries_give_no_rows():
    assert charts.rows_from_entries([], 2026) == []


def test_clean_title_strips_suffixes():
    assert collect.clean_title("Song (feat. Someone)") == "Song"
    assert collect.clean_title("Song - Remix") == "Song"
    assert collect.clean_title("Plain Song") == "Plain Song"


def test_missing_data_gives_meaningful_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(collect, "TRACKS_DIR", tmp_path / "nope")
    assert collect.main(["--market", "US"]) == 2
    assert "No track lists" in capsys.readouterr().err


def test_discover_billboard_only_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "TRACKS_DIR", tmp_path)
    for n in ("tracks_BB_2023.csv", "tracks_US_2023.csv", "tracks_GB_2023.csv"):
        (tmp_path / n).write_text("x", encoding="utf-8")
    assert {s for _, s, _ in collect.discover_track_files("US")} == {"spotify"}
    assert {s for _, s, _ in collect.discover_track_files("US", include_billboard=True)} == {
        "spotify", "billboard"}
    assert {s for _, s, _ in collect.discover_track_files("GB", include_billboard=True)} == {"spotify"}


def test_get_token_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GENIUS_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GENIUS_ACCESS", raising=False)
    f = tmp_path / ".env"
    f.write_text("# c\nOTHER=1\ngenius_access = 'abc123'\n", encoding="utf-8")
    assert collect.get_token(f) == "abc123"


def test_lrclib_success_uses_plain_lyrics():
    s = FakeSession([FakeResp(200, {"plainLyrics": "line one\nline two"})])
    text, reason = collect.fetch_lrclib(s, "Artist", "Song")
    assert text == "line one\nline two" and reason == ""
    assert s.calls[0][1] == {"artist_name": "Artist", "track_name": "Song"}


def test_lrclib_404_is_not_found():
    assert collect.fetch_lrclib(FakeSession([FakeResp(404)]), "A", "S") == (None, "lrclib 404")


def test_lrclib_instrumental_and_empty():
    assert collect.fetch_lrclib(FakeSession([FakeResp(200, {"instrumental": True})]), "A", "S")[1] == "instrumental"
    assert collect.fetch_lrclib(FakeSession([FakeResp(200, {"plainLyrics": None})]), "A", "S")[0] is None


def test_lrclib_retries_on_429(monkeypatch):
    monkeypatch.setattr(collect.time, "sleep", lambda *_: None)
    s = FakeSession([FakeResp(429), FakeResp(200, {"plainLyrics": "ok"})])
    assert collect.fetch_lrclib(s, "A", "S")[0] == "ok"


def test_genius_fallback_disables_on_403(capsys):
    g = collect.GeniusFallback("tok")

    class Boom:
        def search_song(self, *a):
            raise RuntimeError("403 Client Error: Forbidden")

    g.client = Boom()
    assert g.fetch("A", "S") is None and g.disabled
    assert "disabled" in capsys.readouterr().out
    assert g.fetch("A", "S") is None  # stays skipped, no new calls


def test_genius_fallback_without_token_is_disabled():
    assert collect.GeniusFallback(None).fetch("A", "S") is None


def test_end_to_end_logs_found_and_not_found(tmp_path, monkeypatch):
    tracks, lyrics = tmp_path / "tracks", tmp_path / "lyrics"
    tracks.mkdir()
    (tracks / "tracks_US_2023.csv").write_text(
        "year,market,artist,track,popularity\n2023,US,A,Hit,5\n2023,US,B,Unknown,3\n", encoding="utf-8")
    monkeypatch.setattr(collect, "TRACKS_DIR", tracks)
    monkeypatch.setattr(collect, "LYRICS_DIR", lyrics)
    monkeypatch.setattr(collect.time, "sleep", lambda *_: None)

    def fake(session, artist, track, retries=3):
        return ("some words here", "") if track == "Hit" else (None, "lrclib 404")

    monkeypatch.setattr(collect, "fetch_lrclib", fake)
    assert collect.main(["--market", "US", "--years", "2023", "--no-genius"]) == 0
    got = (lyrics / "lyrics_US_2023.csv").read_text(encoding="utf-8")
    assert "Hit" in got and "lrclib" in got and "Unknown" not in got
    assert "Unknown" in (lyrics / "not_found_US_2023.csv").read_text(encoding="utf-8")
    # resumable: second run does not duplicate
    assert collect.main(["--market", "US", "--years", "2023", "--no-genius"]) == 0
    assert (lyrics / "lyrics_US_2023.csv").read_text(encoding="utf-8").count("Hit") == 1
