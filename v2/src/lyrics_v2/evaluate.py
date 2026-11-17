"""Quantitative evaluation for topic models: c_v coherence, topic diversity, and k-scan.

    python -m lyrics_v2.evaluate --market US GB
    python -m lyrics_v2.evaluate --market US GB --scan-k

Reads  data/processed/clean_<market>.csv
       v2/results/models/topics_<market>_<model>.csv
Writes v2/results/eval/metrics_<market>.csv
       v2/results/eval/k_scan_<market>.csv
       v2/results/eval/k_scan_<market>.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .model import ALL_MODELS, fit_lda, fit_nmf
from .paths import PROCESSED_DIR, RESULTS_DIR


def compute_diversity(topics_words, top_n=10):
    """Fraction of unique words among top-N words across all topics (0..1)."""
    sliced = [list(words)[:top_n] for words in topics_words if words]
    total = sum(len(w) for w in sliced)
    if total == 0:
        return 0.0
    unique = len({w for words in sliced for w in words})
    return round(unique / total, 4)


def compute_coherence(topics_words, tokenized_docs, coherence="c_v"):
    """Compute gensim c_v coherence for a list of topic word lists."""
    from gensim import corpora
    from gensim.models import CoherenceModel

    id2word = corpora.Dictionary(tokenized_docs)
    vocab = id2word.token2id
    filtered = [[w for w in words if w in vocab] for words in topics_words]
    filtered = [words for words in filtered if len(words) >= 2]
    if not filtered:
        return 0.0
    cm = CoherenceModel(
        topics=filtered,
        texts=tokenized_docs,
        dictionary=id2word,
        coherence=coherence,
        processes=1,
    )
    val = cm.get_coherence()
    return round(float(val), 4) if val == val else 0.0  # guard NaN


def _load_clean(market, processed_dir=None):
    pdir = Path(processed_dir or PROCESSED_DIR)
    path = pdir / f"clean_{market.upper()}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data missing: {path}. Run `python -m lyrics_v2.clean --market {market}` first."
        )
    df = pd.read_csv(path).dropna(subset=["tokens"])
    return [d.split() for d in df["tokens"].astype(str)]


def evaluate_market(market, models=ALL_MODELS, processed_dir=None, models_dir=None, eval_dir=None):
    tokenized = _load_clean(market, processed_dir)
    mdir = Path(models_dir or (RESULTS_DIR / "models"))
    edir = Path(eval_dir or (RESULTS_DIR / "eval"))
    edir.mkdir(parents=True, exist_ok=True)

    summary_path = mdir / f"model_summary_{market.upper()}.csv"
    outliers_map = {}
    if summary_path.exists():
        sdf = pd.read_csv(summary_path)
        outliers_map = dict(zip(sdf["model"], sdf["outliers"]))

    rows = []
    for name in models:
        tpath = mdir / f"topics_{market.upper()}_{name}.csv"
        if not tpath.exists():
            continue
        tdf = pd.read_csv(tpath)
        topics_words = [
            [w.strip() for w in str(s).split(",") if w.strip()]
            for s in tdf["top_words"].dropna()
        ]
        rows.append({
            "model": name,
            "market": market.upper(),
            "num_topics": len(topics_words),
            "outliers": int(outliers_map.get(name, 0)),
            "c_v": compute_coherence(topics_words, tokenized),
            "topic_diversity": compute_diversity(topics_words),
        })
    out_df = pd.DataFrame(rows)
    out_path = edir / f"metrics_{market.upper()}.csv"
    out_df.to_csv(out_path, index=False)
    return out_df


def scan_k(market, k_values=range(5, 41, 5), passes=30, processed_dir=None, eval_dir=None):
    tokenized = _load_clean(market, processed_dir)
    edir = Path(eval_dir or (RESULTS_DIR / "eval"))
    edir.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in k_values:
        for name, fn in (("lda", fit_lda), ("nmf", fit_nmf)):
            res = fn(tokenized, market.upper(), num_topics=k, passes=passes)
            words = list(res.topics.values())
            rows.append({
                "model": name,
                "market": market.upper(),
                "k": int(k),
                "c_v": compute_coherence(words, tokenized),
                "topic_diversity": compute_diversity(words),
            })
    df = pd.DataFrame(rows)
    df.to_csv(edir / f"k_scan_{market.upper()}.csv", index=False)
    plot_k_scan(df, market.upper(), edir / f"k_scan_{market.upper()}.png")
    return df


def plot_k_scan(df, market, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, grp in df.groupby("model"):
        grp = grp.sort_values("k")
        ax.plot(grp["k"], grp["c_v"], marker="o", label=name.upper())
        best = grp.loc[grp["c_v"].idxmax()]
        ax.annotate(
            f"k={int(best['k'])} ({best['c_v']:.3f})",
            xy=(best["k"], best["c_v"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_title(f"{market} — Coherence (c_v) by Number of Topics (k)")
    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel("c_v coherence")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate topic models (c_v coherence & diversity)")
    ap.add_argument("--market", nargs="+", default=["US"], help="US, GB, ...")
    ap.add_argument("--scan-k", action="store_true", help="Run k=5..40 (step 5) scan for LDA/NMF")
    ap.add_argument("--k-step", type=int, default=5)
    args = ap.parse_args(argv)

    for m in args.market:
        try:
            mdf = evaluate_market(m.upper())
            print(mdf.to_string(index=False))
            if args.scan_k:
                kdf = scan_k(m.upper(), k_values=range(5, 41, args.k_step))
                best = kdf.loc[kdf.groupby("model")["c_v"].idxmax()]
                print(f"[{m.upper()}] best k:\n{best.to_string(index=False)}")
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
