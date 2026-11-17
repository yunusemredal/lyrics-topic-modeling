"""Period (2016-2022 vs 2023-2025) & country (US vs GB) comparison and plots.

    python -m lyrics_v2.compare

Reads  data/processed/clean_{US_2016_2022,US,GB}.csv
       v2/results/models/topics_over_time_{US_2016_2022,US,GB}_bertopic.csv
       v2/results/eval/metrics_{US_2016_2022,US,GB}.csv
Writes v2/results/compare/period_comparison_US.csv
       v2/results/compare/country_comparison_2023_2025.csv
       v2/results/compare/genre_proxy_summary.csv
       v2/results/compare/all_metrics_summary.csv
       v2/results/compare/topics_over_time_US_2016_2025.png
       v2/results/compare/model_comparison.png
       v2/results/compare/wordcloud_{US_2016_2022,US_2023_2025,GB_2023_2025}.png
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .paths import PROCESSED_DIR, RESULTS_DIR

HIPHOP_SLANG_PROXY = {"fuck", "bitch", "shit", "nigga", "ass", "money", "flex", "drip"}
ROMANCE_POP_PROXY = {"love", "heart", "kiss", "dream", "forever", "darling"}


def _word_rates_per_1k(tokens_series):
    counts = Counter()
    total = 0
    for doc in tokens_series.dropna().astype(str):
        words = doc.split()
        counts.update(words)
        total += len(words)
    if total == 0:
        return {}
    return {w: (c / total) * 1000.0 for w, c in counts.items()}


def compare_word_rates(df_a, df_b, label_a="2016_2022", label_b="2023_2025", min_rate=0.5):
    """Compare token frequencies (per 1,000 tokens) between two cleaned DataFrames."""
    ra = _word_rates_per_1k(df_a["tokens"])
    rb = _word_rates_per_1k(df_b["tokens"])
    rows = []
    for w in set(ra) | set(rb):
        va, vb = ra.get(w, 0.0), rb.get(w, 0.0)
        if max(va, vb) < min_rate:
            continue
        rows.append({
            "word": w,
            f"per_1k_{label_a}": round(va, 3),
            f"per_1k_{label_b}": round(vb, 3),
            "diff_per_1k": round(vb - va, 3),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("diff_per_1k", ascending=False).reset_index(drop=True)


def compute_genre_proxy(df, slice_name):
    """Lexical proxy for hip-hop/explicit vs romance/pop share (genre metadata is empty)."""
    rows = []
    for year, grp in df.groupby("year"):
        docs = [set(str(t).split()) for t in grp["tokens"].dropna()]
        n = len(docs)
        if n == 0:
            continue
        hh = sum(1 for d in docs if d & HIPHOP_SLANG_PROXY) / n
        pop = sum(1 for d in docs if d & ROMANCE_POP_PROXY) / n
        rows.append({
            "slice": slice_name,
            "year": int(year),
            "docs": n,
            "hiphop_proxy_share": round(hh, 4),
            "romance_proxy_share": round(pop, 4),
        })
    return pd.DataFrame(rows)


def plot_wordcloud(tokens_series, title, out_path):
    from wordcloud import WordCloud

    text = " ".join(tokens_series.dropna().astype(str).tolist())
    if not text.strip():
        return
    wc = WordCloud(
        width=900, height=500, background_color="white",
        max_words=80, collocations=False, random_state=42,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_topics_over_time(tot_v1, tot_v2, out_path):
    """Plot top BERTopic themes over time across US 2016-2022 and US 2023-2025."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    specs = [
        (axes[0], tot_v1, "US 2016–2022 (Pandemic 2020–2021 shaded)", True),
        (axes[1], tot_v2, "US 2023–2025", False),
    ]
    for ax, df, title, shade_covid in specs:
        sub = df[df["Topic"] != -1].copy()
        top_labels = (
            sub.groupby("Label")["Frequency"].sum()
            .sort_values(ascending=False).head(5).index.tolist()
        )
        piv = (
            sub[sub["Label"].isin(top_labels)]
            .pivot_table(index="Timestamp", columns="Label", values="Frequency", aggfunc="sum", fill_value=0)
        )
        for col in piv.columns:
            ax.plot(piv.index, piv[col], marker="o", linewidth=2, label=col)
        if shade_covid:
            ax.axvspan(2020, 2021, color="gray", alpha=0.15, label="2020–2021")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("Track count")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("US Top Lyrics Themes Over Time (BERTopic)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(summary_df, out_path):
    """Grouped bar chart of c_v coherence and topic diversity across all 5 models."""
    order = ["lda", "nmf", "top2vec", "bertopic", "fastopic"]
    slices = ["US_2016_2022", "US", "GB"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title in (
        (axes[0], "c_v", "Topic Coherence (c_v)"),
        (axes[1], "topic_diversity", "Topic Diversity"),
    ):
        piv = summary_df.pivot(index="model", columns="market", values=metric)
        piv = piv.reindex([m for m in order if m in piv.index])
        piv = piv[[s for s in slices if s in piv.columns]]
        piv.plot(kind="bar", ax=ax, width=0.75)
        ax.set_title(title)
        ax.set_xlabel("Model (2003 → 2024)")
        ax.set_ylabel(metric)
        ax.set_xticklabels([m.upper() for m in piv.index], rotation=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Slice", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_compare(processed_dir=None, results_dir=None):
    pdir = Path(processed_dir or PROCESSED_DIR)
    rdir = Path(results_dir or RESULTS_DIR)
    cdir = rdir / "compare"
    cdir.mkdir(parents=True, exist_ok=True)

    required = {
        "US_2016_2022": pdir / "clean_US_2016_2022.csv",
        "US": pdir / "clean_US.csv",
        "GB": pdir / "clean_GB.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing cleaned files: " + ", ".join(missing))

    dfs = {k: pd.read_csv(p) for k, p in required.items()}

    # 1. Period & country word shifts
    period_df = compare_word_rates(dfs["US_2016_2022"], dfs["US"], "2016_2022", "2023_2025")
    period_df.to_csv(cdir / "period_comparison_US.csv", index=False)

    country_df = compare_word_rates(dfs["US"], dfs["GB"], "US", "GB")
    country_df.to_csv(cdir / "country_comparison_2023_2025.csv", index=False)

    # 2. Genre proxy confounder
    proxy_df = pd.concat(
        [compute_genre_proxy(dfs[k], k) for k in ("US_2016_2022", "US", "GB")],
        ignore_index=True,
    )
    proxy_df.to_csv(cdir / "genre_proxy_summary.csv", index=False)

    # 3. Wordclouds
    plot_wordcloud(dfs["US_2016_2022"]["tokens"], "US 2016–2022 (Top Lyrics Tokens)", cdir / "wordcloud_US_2016_2022.png")
    plot_wordcloud(dfs["US"]["tokens"], "US 2023–2025 (Top Lyrics Tokens)", cdir / "wordcloud_US_2023_2025.png")
    plot_wordcloud(dfs["GB"]["tokens"], "GB 2023–2025 (Top Lyrics Tokens)", cdir / "wordcloud_GB_2023_2025.png")

    # 4. Combined metrics & plot
    edir = rdir / "eval"
    metric_frames = [
        pd.read_csv(edir / f"metrics_{k}.csv")
        for k in ("US_2016_2022", "US", "GB")
        if (edir / f"metrics_{k}.csv").exists()
    ]
    if metric_frames:
        all_metrics = pd.concat(metric_frames, ignore_index=True)
        all_metrics.to_csv(cdir / "all_metrics_summary.csv", index=False)
        plot_model_comparison(all_metrics, cdir / "model_comparison.png")

    # 5. Topics-over-time plot
    mdir = rdir / "models"
    tot1, tot2 = mdir / "topics_over_time_US_2016_2022_bertopic.csv", mdir / "topics_over_time_US_bertopic.csv"
    if tot1.exists() and tot2.exists():
        plot_topics_over_time(pd.read_csv(tot1), pd.read_csv(tot2), cdir / "topics_over_time_US_2016_2025.png")

    return cdir


def main(argv=None):
    argparse.ArgumentParser(description="Compare periods and markets; generate plots").parse_args(argv)
    try:
        out = run_compare()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"Comparison artifacts written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
