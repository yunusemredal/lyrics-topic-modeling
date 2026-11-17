from lyrics_v2 import collect

HEADER = "year,market,artist,track,popularity\n"


def test_zero_popularity_falls_back_to_file_order(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(HEADER + "2023,US,A,x,0\n2023,US,B,y,0\n", encoding="utf-8")
    assert [r["popularity"] for r in collect.read_tracks(p)] == [2, 1]


def test_real_popularity_is_kept(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text(HEADER + "2023,US,A,x,70\n2023,US,B,y,50\n", encoding="utf-8")
    assert [r["popularity"] for r in collect.read_tracks(p)] == ["70", "50"]

