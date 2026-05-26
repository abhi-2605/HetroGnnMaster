"""
Generate thesis-quality figures from existing result artifacts.

Reads from result folders (no model evaluation here) and writes both
.pdf and .png copies into results/plots/. Each figure is self-contained
and captioned in code so the corresponding LaTeX caption can be lifted
directly.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Global styling — thesis-friendly defaults
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colour palette — colourblind-safe, print-friendly
C_TGNN       = "#1f77b4"
C_TGNN_DARK  = "#0b3a64"
C_RF         = "#d62728"
C_XGB        = "#ff7f0e"
C_STATIC_GAT = "#9467bd"
C_STATIC_GCN = "#8c564b"
C_BENIGN     = "#2ca02c"
C_RANSOM     = "#d62728"
C_MALWARE    = "#7f7f7f"

CLASS_COLORS = {
    "Ransomware": C_RANSOM,
    "Benign":     C_BENIGN,
    "Other Malwares": C_MALWARE,
    "Other Malware":  C_MALWARE,
}

BRANCH_COLORS = {
    "semantic":   "#4c72b0",
    "structural": "#dd8452",
    "temporal":   "#55a868",
}


def _savefig(fig: plt.Figure, name: str) -> None:
    png = PLOTS_DIR / f"{name}.png"
    fig.savefig(png)
    plt.close(fig)
    print(f"  ✓ {png.name}")


def _annot_bars(ax, bars, fmt="{:.2f}", fontsize=8, offset=2.0, color="#222222"):
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=fontsize, color=color,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Overall performance on zero-day evaluation set
# ═════════════════════════════════════════════════════════════════════════════

def fig1_overall_performance() -> None:
    print("[fig1] Overall performance — RF / XGB / TGNN on zero-day eval")

    custom = pd.read_csv(PROJECT_ROOT / "baselines_custom_eval" / "classical_custom_eval_results.csv")
    custom = custom[custom["model"].str.contains("custom_eval")]
    custom = custom.rename(columns={"model": "system"})

    tgnn = json.load(open(PROJECT_ROOT / "results" / "custom_eval" / "zero_day_results.json"))
    tgnn_3c = tgnn["metrics_3class"]
    tgnn_tvb = tgnn["metrics_threat_vs_benign"]
    tgnn_row = {
        "system": "Proposed TGNN (custom_eval)",
        "macro_f1": tgnn_3c["macro_f1"],
        "ransom_R": tgnn_3c["per_class_recall"][0],
        "threat_R": tgnn_tvb["threat_recall"],
        "threats_missed": tgnn_tvb["threats_missed"],
        "accuracy": tgnn_3c["accuracy"],
    }
    rows = []
    for _, r in custom.iterrows():
        name = r["system"].replace(" (custom_eval)", "")
        rows.append({
            "system": name,
            "macro_f1": r["macro_f1"],
            "ransom_R": r["ransom_R"],
            "threat_R": (r["ransom_R"] * 75 + r["malware_R"] * 200) / 275.0,  # weighted by support
            "threats_missed": r["threats_missed"],
            "accuracy": r["accuracy"],
        })
    rows.append({
        "system": "Proposed TGNN",
        "macro_f1": tgnn_row["macro_f1"],
        "ransom_R": tgnn_row["ransom_R"],
        "threat_R": tgnn_row["threat_R"],
        "threats_missed": tgnn_row["threats_missed"],
        "accuracy": tgnn_row["accuracy"],
    })
    df = pd.DataFrame(rows).set_index("system").reindex(
        ["Random Forest", "XGBoost", "KNN", "Proposed TGNN"]
    ).dropna(how="all")

    colors = {"Random Forest": C_RF, "XGBoost": C_XGB, "KNN": "#8c564b", "Proposed TGNN": C_TGNN}
    bar_colors = [colors.get(s, "#888888") for s in df.index]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))

    ax = axes[0]
    bars = ax.bar(df.index, df["macro_f1"], color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1 (3-class)")
    ax.set_title("Macro-F1")
    _annot_bars(ax, bars, fmt="{:.3f}")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    width = 0.36
    x = np.arange(len(df))
    b1 = ax.bar(x - width/2, df["ransom_R"], width, label="Ransomware recall",
                color=[colors.get(s, "#888888") for s in df.index],
                edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + width/2, df["threat_R"], width, label="Threat recall (Ransom + Malware)",
                color="white", edgecolor=[colors.get(s, "#888888") for s in df.index],
                linewidth=1.4, hatch="///")
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=20)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Recall")
    ax.set_title("Recall — ransomware vs. all threats")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2, frameon=False)
    _annot_bars(ax, b1, fmt="{:.2f}")
    _annot_bars(ax, b2, fmt="{:.2f}")

    ax = axes[2]
    bars = ax.bar(df.index, df["threats_missed"], color=bar_colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("# threats missed (FN)")
    ax.set_title("Threats missed — lower is better")
    _annot_bars(ax, bars, fmt="{:.0f}")
    ax.tick_params(axis="x", rotation=20)
    ymax = max(df["threats_missed"].max() * 1.25, 5)
    ax.set_ylim(0, ymax)

    fig.suptitle(
        "Zero-day evaluation (475 samples: 75 ransomware, 200 benign, 200 other-malware)",
        fontsize=11.5, y=1.04,
    )
    fig.tight_layout()
    _savefig(fig, "fig1_overall_performance")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Dataset characterization
# ═════════════════════════════════════════════════════════════════════════════

def fig2_dataset_characterization() -> None:
    print("[fig2] Dataset characterization — graph size + benign mix")

    api = pd.read_csv(PROJECT_ROOT / "results" / "api_embedding_check" / "api_embedding_check_split_val.csv")
    n_api = api["num_api_nodes"].dropna().astype(int)

    benign = pd.read_csv(PROJECT_ROOT / "benign_operation_report" / "benign_split_summary.csv")
    pivot = benign.pivot_table(
        index="benign_op_type", columns="split", values="count", aggfunc="sum"
    ).fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    bins = np.logspace(np.log10(max(n_api.min(), 1)), np.log10(n_api.max() + 1), 40)
    ax.hist(n_api, bins=bins, color=C_TGNN, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("API nodes per behavioural graph (log scale)")
    ax.set_ylabel("Number of graphs")
    ax.set_title(f"Behavioural graph size (val split, n={len(n_api)})")
    median = n_api.median()
    mean = n_api.mean()
    ax.axvline(median, color="black", linestyle="--", linewidth=0.9, label=f"median={int(median):,}")
    ax.axvline(mean,   color=C_RANSOM, linestyle=":", linewidth=1.1, label=f"mean={int(mean):,}")
    ax.legend(loc="upper right")

    ax = axes[1]
    split_order = [s for s in ["train", "val", "test"] if s in pivot.columns]
    colors = ["#4c72b0", "#dd8452", "#55a868"][: len(split_order)]
    pivot[split_order].plot(
        kind="barh", stacked=True, ax=ax, color=colors,
        edgecolor="black", linewidth=0.4,
    )
    ax.set_xlabel("Number of benign samples")
    ax.set_ylabel("")
    ax.set_title("Benign category composition")
    ax.legend(title="Split", loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Dataset characterization — heavy-tailed graph sizes and benign-software diversity",
        fontsize=11.5, y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, "fig2_dataset_characterization")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Static vs Temporal models under structural attacks
# ═════════════════════════════════════════════════════════════════════════════

def fig3_static_vs_temporal() -> None:
    print("[fig3] Static GCN/GAT vs proposed TGNN — clean + structural attacks")

    s = json.load(open(PROJECT_ROOT / "results" / "static_vs_existing_tgnn_strict" / "comparison_summary.json"))
    models = list(s["results"].keys())
    tracks = ["clean", "bigram_attack", "structure_attack", "combined_attack"]

    rows = []
    for m in models:
        for t in tracks:
            if t not in s["results"][m]:
                continue
            metrics = s["results"][m][t]["metrics"]
            tvb = s["results"][m][t].get("tvb", {})
            rows.append({
                "model": m, "track": t,
                "macro_f1": metrics["macro_f1"],
                "threats_missed": tvb.get("threats_missed", np.nan),
                "threat_recall": tvb.get("threat_recall", np.nan),
            })
    df = pd.DataFrame(rows)

    def short(m):
        if "GCN" in m: return "Static GCN"
        if "GAT" in m: return "Static GAT"
        return "Proposed TGNN"
    df["model_short"] = df["model"].map(short)

    pretty = {
        "clean": "Clean",
        "bigram_attack": "Bigram interleaving",
        "structure_attack": "Structural removal",
        "combined_attack": "Combined attack",
    }
    df["track_pretty"] = df["track"].map(pretty)

    model_order = ["Static GCN", "Static GAT", "Proposed TGNN"]
    track_order = [pretty[t] for t in tracks if pretty[t] in df["track_pretty"].unique()]
    f1_mat = df.pivot(index="model_short", columns="track_pretty", values="macro_f1").reindex(model_order)[track_order]
    miss_mat = df.pivot(index="model_short", columns="track_pretty", values="threats_missed").reindex(model_order)[track_order]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    x = np.arange(len(track_order))
    width = 0.26
    model_colors = {"Static GCN": C_STATIC_GCN, "Static GAT": C_STATIC_GAT, "Proposed TGNN": C_TGNN}
    for i, m in enumerate(model_order):
        if m not in f1_mat.index:
            continue
        bars = ax.bar(x + (i - 1) * width, f1_mat.loc[m].values, width,
                      color=model_colors[m], edgecolor="black", linewidth=0.5,
                      label=m)
        _annot_bars(ax, bars, fmt="{:.2f}", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(track_order, rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1 (3-class)")
    ax.set_title("Macro-F1 across evaluation tracks")

    ax = axes[1]
    for i, m in enumerate(model_order):
        if m not in miss_mat.index:
            continue
        bars = ax.bar(x + (i - 1) * width, miss_mat.loc[m].values, width,
                      color=model_colors[m], edgecolor="black", linewidth=0.5,
                      label=m)
        _annot_bars(ax, bars, fmt="{:.0f}", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(track_order, rotation=12)
    ax.set_ylabel("# threats missed (FN, 275 threats total)")
    ax.set_title("Threats missed — lower is better")
    ymax = miss_mat.values.max() * 1.30 if np.isfinite(miss_mat.values.max()) else 50
    ax.set_ylim(0, max(ymax, 10))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncols=len(model_order), bbox_to_anchor=(0.5, -0.03),
               frameon=False)

    fig.suptitle(
        "Static GNN baselines vs proposed TGNN under structural perturbations "
        "(custom-eval, 475 samples)",
        fontsize=11.5, y=1.03,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _savefig(fig, "fig3_static_vs_temporal")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 4 — Dual-attack robustness vs classical TF-IDF baselines
# ═════════════════════════════════════════════════════════════════════════════

def fig4_robustness_dual_attack() -> None:
    print("[fig4] Dual-attack robustness — TGNN vs RF/XGB feature variants")

    j = json.load(open(PROJECT_ROOT / "robustness_compare_dual" / "comparison_summary.json"))

    tracks = ["clean", "unigram_attack", "bigram_attack", "combined_attack"]
    pretty = {
        "clean": "Clean",
        "unigram_attack": "Unigram\nattack",
        "bigram_attack":  "Bigram\nattack",
        "combined_attack":"Combined\nattack",
    }

    rows = []
    for fam in ("random_forest", "xgboost"):
        for tf in ("unigram", "unigram_bigram", "bigram"):
            b = j["baseline"][fam][tf]
            for tr in tracks:
                cev = b.get(f"{tr}_custom_eval" if tr != "clean" else "clean_custom_eval", {})
                tvb = b.get(f"{tr}_tvb" if tr != "clean" else "clean_tvb", {})
                rows.append({
                    "system": f"{'RF' if fam=='random_forest' else 'XGB'} ({tf})",
                    "track": tr,
                    "macro_f1": cev.get("macro_f1", np.nan),
                    "threat_recall": tvb.get("threat_recall", np.nan),
                    "threats_missed": tvb.get("threats_missed", np.nan),
                })
    for tr in tracks:
        cev = j["tgnn"].get(f"{tr}_custom_eval" if tr != "clean" else "clean_custom_eval", {})
        tvb = j["tgnn"].get(f"{tr}_tvb" if tr != "clean" else "clean_tvb", {})
        rows.append({
            "system": "Proposed TGNN",
            "track": tr,
            "macro_f1": cev.get("macro_f1", np.nan),
            "threat_recall": tvb.get("threat_recall", np.nan),
            "threats_missed": tvb.get("threats_missed", np.nan),
        })

    df = pd.DataFrame(rows)
    system_order = [
        "RF (unigram)", "RF (unigram_bigram)", "RF (bigram)",
        "XGB (unigram)", "XGB (unigram_bigram)", "XGB (bigram)",
        "Proposed TGNN",
    ]
    f1_mat = df.pivot(index="system", columns="track", values="macro_f1").reindex(system_order)[tracks]
    tr_mat = df.pivot(index="system", columns="track", values="threat_recall").reindex(system_order)[tracks]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})

    ax = axes[0]
    im = ax.imshow(f1_mat.values, cmap="RdYlGn", vmin=0.15, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(tracks))); ax.set_xticklabels([pretty[t] for t in tracks])
    ax.set_yticks(range(len(system_order))); ax.set_yticklabels(system_order)
    for i in range(f1_mat.shape[0]):
        for k in range(f1_mat.shape[1]):
            v = f1_mat.values[i, k]
            if np.isfinite(v):
                ax.text(k, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if v > 0.55 else "white")
    ax.set_title("Macro-F1 under each attack track")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Macro-F1", rotation=270, labelpad=12)
    ax.grid(False)

    ax = axes[1]
    markers = ["o", "s", "D", "^", "v", "P", "*"]
    colors = [C_RF, C_RF, C_RF, C_XGB, C_XGB, C_XGB, C_TGNN]
    linestyles = ["-", "--", ":", "-", "--", ":", "-"]
    for i, s in enumerate(system_order):
        ax.plot(range(len(tracks)), tr_mat.loc[s].values,
                marker=markers[i], linestyle=linestyles[i],
                color=colors[i], lw=2 if s == "Proposed TGNN" else 1.2,
                markersize=8 if s == "Proposed TGNN" else 6,
                markeredgecolor="black", markeredgewidth=0.4,
                label=s)
    ax.set_xticks(range(len(tracks))); ax.set_xticklabels([pretty[t] for t in tracks])
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Threat recall (Ransomware + Other Malware)")
    ax.set_title("Threat recall preservation under attack")
    ax.legend(loc="lower left", fontsize=8, ncols=2)
    ax.axhline(0.85, color="#888888", lw=0.8, linestyle="--")
    ax.text(0.02, 0.86, "0.85 floor", color="#666666", fontsize=8, transform=ax.get_yaxis_transform())

    fig.suptitle(
        "Robustness to API renaming attacks — classical TF-IDF baselines vs proposed TGNN",
        fontsize=11.5, y=1.02,
    )
    fig.tight_layout()
    _savefig(fig, "fig4_robustness_dual_attack")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 5 — Semantic vs random substitution
# ═════════════════════════════════════════════════════════════════════════════

def fig5_semantic_substitution() -> None:
    print("[fig5] Semantic-aware vs random API substitution")

    base = PROJECT_ROOT / "results" / "semantic_vector_substitution_test"
    splits = ["train", "val", "test"]
    rows = []
    for sp in splits:
        path = base / f"semantic_vector_substitution_{sp}.summary.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        rows.append({
            "split": sp,
            "n_samples": d["n_clean"],
            "clean_f1": d["clean_metrics"]["macro_f1"],
            "semantic_sub_f1": d["semantic_substitution_metrics"]["macro_f1"],
            "random_sub_f1":   d["random_substitution_metrics"]["macro_f1"],
            "semantic_flip": d["semantic_substitution_comparison"]["prediction_flip_rate"],
            "random_flip":   d["random_substitution_comparison"]["prediction_flip_rate"],
            "semantic_t2b": d["semantic_substitution_comparison"]["threat_to_benign_flip_rate"],
            "random_t2b":   d["random_substitution_comparison"]["threat_to_benign_flip_rate"],
            "semantic_conf_drop": d["semantic_substitution_comparison"]["mean_confidence_drop_on_clean_pred"],
            "random_conf_drop":   d["random_substitution_comparison"]["mean_confidence_drop_on_clean_pred"],
            "rename_frac": d["semantic_substitution_comparison"]["mean_realized_sub_fraction"],
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))

    width = 0.27
    x = np.arange(len(df))
    ax = axes[0]
    b0 = ax.bar(x - width, df["clean_f1"], width, label="Clean",
                color="#999999", edgecolor="black", linewidth=0.5)
    b1 = ax.bar(x,         df["semantic_sub_f1"], width, label="Semantic substitution",
                color=C_TGNN, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width, df["random_sub_f1"], width, label="Random substitution",
                color=C_XGB, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(df["split"], df["n_samples"])])
    ax.set_ylim(0, 1.18); ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 — clean vs perturbed")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=3, frameon=False)
    for bars in (b0, b1, b2):
        _annot_bars(ax, bars, fmt="{:.2f}", fontsize=8)

    ax = axes[1]
    width = 0.36
    b1 = ax.bar(x - width/2, df["semantic_flip"] * 100, width,
                label="Semantic", color=C_TGNN, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width/2, df["random_flip"] * 100, width,
                label="Random", color=C_XGB, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(df["split"])
    ax.set_ylabel("Prediction flip rate (%)")
    ax.set_title("Prediction flip rate")
    ax.legend(loc="upper right")
    _annot_bars(ax, b1, fmt="{:.1f}", fontsize=8)
    _annot_bars(ax, b2, fmt="{:.1f}", fontsize=8)
    ax.set_ylim(0, max(df[["semantic_flip", "random_flip"]].values.max() * 100 * 1.4, 5))

    ax = axes[2]
    b1 = ax.bar(x - width/2, df["semantic_conf_drop"], width,
                label="Semantic", color=C_TGNN, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width/2, df["random_conf_drop"], width,
                label="Random", color=C_XGB, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(df["split"])
    ax.set_ylabel("Mean confidence drop on originally-correct preds")
    ax.set_title("Confidence drop")
    ax.legend(loc="upper right")
    _annot_bars(ax, b1, fmt="{:.3f}", fontsize=8)
    _annot_bars(ax, b2, fmt="{:.3f}", fontsize=8)

    fig.suptitle(
        "Robustness to API renaming — semantically-similar substitution (mean ≈96% APIs renamed) "
        "vs random substitution",
        fontsize=11.5, y=1.04,
    )
    fig.tight_layout()
    _savefig(fig, "fig5_semantic_substitution")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 6 — Component attribution (ablation impact)
# ═════════════════════════════════════════════════════════════════════════════

def fig6_component_attribution() -> None:
    print("[fig6] Component attribution — ablation impact")

    j = json.load(open(PROJECT_ROOT / "results" / "component_attribution_improved" / "attribution_test_summary.json"))
    rows = []
    for name, m in j["by_ablation"].items():
        rows.append({
            "ablation": name,
            "decision_flip": m["decision_flip_rate"],
            "threat_flip":   m["threat_binary_flip_rate"],
            "margin_drop":   m["mean_margin_drop"],
            "conf_drop":     m["mean_pred_conf_drop"],
            "correct_after": m["correct_after_rate"],
            "n":             m["n"],
        })
    df = pd.DataFrame(rows).sort_values("decision_flip", ascending=True)

    def categorise(name: str) -> str:
        if name.startswith("semantic_"):  return "Semantic"
        if name.startswith("temporal_"):  return "Temporal"
        if name.startswith("remove_edge"):return "Structure (edges)"
        if name.startswith("remove_all_structure"): return "Structure (all)"
        if name.startswith("combined_"):  return "Combined"
        return "Other"
    df["category"] = df["ablation"].map(categorise)

    cat_colors = {
        "Semantic":          "#1f77b4",
        "Structure (edges)": "#dd8452",
        "Structure (all)":   "#8c564b",
        "Temporal":          "#55a868",
        "Combined":          "#d62728",
    }

    pretty = {
        "semantic_zero_bert":          "Zero BERT (semantic OFF)",
        "semantic_shuffle_bert":       "Shuffle BERT vectors",
        "semantic_noise_bert":         "Noise BERT vectors",
        "remove_all_structure_keep_temporal": "Remove all non-temporal edges",
        "remove_edge_api_file":        "Remove API→file edges",
        "remove_edge_api_registry":    "Remove API→registry edges",
        "remove_edge_api_network":     "Remove API→network edges",
        "remove_edge_process_api":     "Remove process→API edges",
        "remove_edge_api_temporal":    "Remove API↔API temporal edges",
        "temporal_zero_lstm":          "Disable LSTM temporal branch",
        "temporal_shuffle_timestamps": "Shuffle timestamps",
        "temporal_reverse_timestamps": "Reverse timestamps",
        "temporal_collapse_timestamps":"Collapse timestamps",
        "combined_no_semantic_no_structure": "Combined: no semantic + no structure",
        "combined_no_semantic_no_temporal":  "Combined: no semantic + no temporal",
        "combined_no_structure_no_temporal": "Combined: no structure + no temporal",
    }
    df["label"] = df["ablation"].map(pretty).fillna(df["ablation"])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0),
                             gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.08})

    ax = axes[0]
    colors = [cat_colors[c] for c in df["category"]]
    bars = ax.barh(df["label"], df["decision_flip"] * 100,
                   color=colors, edgecolor="black", linewidth=0.5)
    xmax = max(df["decision_flip"].max() * 100 * 1.32, 5)
    for b, v in zip(bars, df["decision_flip"] * 100):
        ax.text(v + xmax * 0.012, b.get_y() + b.get_height()/2, f"{v:.1f}%",
                va="center", fontsize=8.5)
    ax.set_xlabel("Decision flip rate (% of test graphs)")
    ax.set_title("Effect on 3-class decision")
    ax.set_xlim(0, xmax)

    ax = axes[1]
    bars = ax.barh(df["label"], df["margin_drop"],
                   color=colors, edgecolor="black", linewidth=0.5)
    xmax = df["margin_drop"].max() * 1.30
    for b, v in zip(bars, df["margin_drop"]):
        ax.text(v + xmax * 0.012, b.get_y() + b.get_height()/2, f"{v:.3f}",
                va="center", fontsize=8.5)
    ax.set_xlabel("Mean margin drop (top-1 minus runner-up)")
    ax.set_title("Effect on prediction margin")
    ax.set_xlim(0, xmax)
    ax.set_yticklabels([])

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, edgecolor="black", linewidth=0.5)
               for c in cat_colors.values()]
    fig.legend(handles, list(cat_colors.keys()),
               loc="lower center", ncols=len(cat_colors), bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=10)

    fig.suptitle(
        f"Component attribution on test split (n={df['n'].iloc[0]} graphs per ablation, "
        f"4 800 total ablation runs)",
        fontsize=11.5, y=1.00,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    _savefig(fig, "fig6_component_attribution")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 7 — Adaptive branch usage by class
# ═════════════════════════════════════════════════════════════════════════════

def fig7_branch_usage() -> None:
    print("[fig7] Adaptive branch fusion — semantic / structural / temporal usage")

    j = json.load(open(PROJECT_ROOT / "results" / "custom_eval" / "zero_day_results.json"))
    bu = j["branch_usage_summary"]
    by_true = bu["by_true_class"]
    overall = bu["overall_mean_weights"]
    classes = ["Ransomware", "Benign", "Other Malwares"]

    weights = {c: by_true[c] for c in classes}
    dominant = {c: by_true[c]["dominant_branch_counts"] for c in classes}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    branch_order = ["semantic", "structural", "temporal"]
    bottoms = np.zeros(len(classes))
    for b in branch_order:
        vals = np.array([weights[c][f"{b}_mean"] for c in classes])
        bars = ax.bar(classes, vals, bottom=bottoms,
                      color=BRANCH_COLORS[b], edgecolor="black", linewidth=0.5,
                      label=b.capitalize())
        for x, (v, btm) in enumerate(zip(vals, bottoms)):
            if v > 0.07:
                ax.text(x, btm + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
        bottoms += vals
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean gate weight")
    ax.set_title("Mean adaptive-fusion weights by true class")
    for i, c in enumerate(classes):
        n = by_true[c]["count"]
        ax.text(i, -0.06, f"n={n}", ha="center", va="top", fontsize=8.5,
                color="#555555", transform=ax.get_xaxis_transform())

    ax = axes[1]
    width = 0.27
    x = np.arange(len(classes))
    for i, b in enumerate(branch_order):
        vals = np.array([dominant[c].get(b, 0) for c in classes], dtype=float)
        denom = np.array([by_true[c]["count"] for c in classes], dtype=float)
        pct = 100 * vals / np.maximum(denom, 1)
        bars = ax.bar(x + (i - 1) * width, pct, width,
                      color=BRANCH_COLORS[b], edgecolor="black", linewidth=0.5,
                      label=b.capitalize())
        _annot_bars(ax, bars, fmt="{:.0f}%", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("% of samples where branch dominates")
    ax.set_title("Dominant branch — distribution by true class")
    ax.set_ylim(0, 110)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Branch",
               loc="lower center", ncols=len(labels), bbox_to_anchor=(0.5, -0.02),
               frameon=False)

    fig.suptitle(
        f"Adaptive branch fusion — overall weights "
        f"sem={overall['semantic']:.2f} / struc={overall['structural']:.2f} / temp={overall['temporal']:.2f}; "
        "ransomware is semantic-dominant, benign is structural-dominant",
        fontsize=11, y=1.02,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    _savefig(fig, "fig7_branch_usage")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 8 — Zero-day distribution shift
# ═════════════════════════════════════════════════════════════════════════════

def fig8_zero_day_distribution_shift() -> None:
    print("[fig8] Zero-day distribution shift — API token shift + node-type stats")

    delta = pd.read_csv(PROJECT_ROOT / "results" / "zero_day_compare" / "api_token_frequency_delta.csv")
    delta = delta.dropna(subset=["abs_freq_delta"])

    top_new = delta[delta["freq_delta_all_minus_old"] > 0].nlargest(10, "abs_freq_delta")
    top_old = delta[delta["freq_delta_all_minus_old"] < 0].nlargest(10, "abs_freq_delta")
    top = pd.concat([top_old.iloc[::-1], top_new], ignore_index=True)
    top["delta_pct"] = top["freq_delta_all_minus_old"] * 100
    top["color"] = top["delta_pct"].apply(lambda v: C_RANSOM if v < 0 else C_TGNN)

    node_df = pd.read_csv(PROJECT_ROOT / "results" / "zero_day_compare" / "node_type_stats_by_folder.csv")
    node_df = node_df[node_df["metric"].str.startswith("nodes_")]
    node_df["nodetype"] = node_df["metric"].str.replace("nodes_", "")

    folder_summary = json.load(open(PROJECT_ROOT / "results" / "zero_day_compare" / "folder_level_summary.json"))
    n_old = folder_summary["num_graphs"]["old"]
    n_all = folder_summary["num_graphs"]["all"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    y = np.arange(len(top))
    bars = ax.barh(y, top["delta_pct"], color=top["color"],
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(top["api_name"])
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Frequency change (percentage points)  —  zero-day  minus  seen")
    ax.set_title("Top API frequency shifts between training-seen and zero-day families")
    for b, v in zip(bars, top["delta_pct"]):
        offset = 0.2 if v >= 0 else -0.2
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, b.get_y() + b.get_height()/2,
                f"{v:+.2f}pp", va="center", ha=ha, fontsize=8.5)
    ax.set_xlim(top["delta_pct"].min() * 1.25, top["delta_pct"].max() * 1.25)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_RANSOM, edgecolor="black", linewidth=0.5),
        plt.Rectangle((0, 0), 1, 1, color=C_TGNN,   edgecolor="black", linewidth=0.5),
    ]
    ax.legend(handles, ["More frequent in seen", "More frequent in zero-day"],
              loc="lower right", fontsize=8.5)

    ax = axes[1]
    nodetypes = ["api", "file", "registry", "process", "network"]
    nodetypes = [nt for nt in nodetypes if nt in node_df["nodetype"].unique()]
    x = np.arange(len(nodetypes))
    width = 0.38

    means_old = []; means_all = []
    for nt in nodetypes:
        sub = node_df[node_df["nodetype"] == nt]
        means_old.append(float(sub[sub["folder"] == "old"]["mean"].values[0]))
        means_all.append(float(sub[sub["folder"] == "all"]["mean"].values[0]))

    b1 = ax.bar(x - width/2, means_old, width,
                label=f"Seen families (n={n_old})", color=C_TGNN, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + width/2, means_all, width,
                label=f"Zero-day families (n={n_all})", color=C_XGB, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(nodetypes)
    ax.set_yscale("log")
    ax.set_ylabel("Mean nodes per graph (log scale)")
    ax.set_title("Mean node-type counts — seen vs zero-day")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, frameon=False)
    ymax_data = max(max(means_old), max(means_all))
    ax.set_ylim(0.7, ymax_data * 4.0)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h <= 0:
                continue
            ax.text(b.get_x() + b.get_width()/2, h * 1.18,
                    f"{h:,.0f}", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle(
        "Zero-day distribution shift — API vocabulary and graph-structure changes that the model must absorb",
        fontsize=11.5, y=1.01,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _savefig(fig, "fig8_zero_day_distribution_shift")


# ═════════════════════════════════════════════════════════════════════════════
#  Figure 9 — Training curves (parsed from results/train.log)
# ═════════════════════════════════════════════════════════════════════════════

def _parse_train_log(path: Path) -> pd.DataFrame:
    """Parse one Train/Val block per epoch from training log."""
    import re

    epoch_re   = re.compile(r"^Epoch (\d+)/\d+")
    train_re   = re.compile(
        r"^Train\s+loss=([\d.]+)\s+macro_f1=([\d.]+).*?"
        r"Ransom P=([\d.]+) R=([\d.]+)\s+"
        r"Benign P=([\d.]+) R=([\d.]+)\s+"
        r"Malware P=([\d.]+) R=([\d.]+)"
    )
    val_re = re.compile(
        r"^Val\s+loss=([\d.]+)\s+macro_f1=([\d.]+).*?"
        r"Ransom P=([\d.]+) R=([\d.]+)\s+"
        r"Benign P=([\d.]+) R=([\d.]+)\s+"
        r"Malware P=([\d.]+) R=([\d.]+)"
    )
    tvb_re = re.compile(
        r"Threat P=([\d.]+) R=([\d.]+) F1=([\d.]+).*?"
        r"Benign P=([\d.]+) R=([\d.]+) F1=([\d.]+).*?"
        r"Acc=([\d.]+)\s+Threats missed=(\d+)"
    )

    rows = []
    cur = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = epoch_re.match(line)
            if m:
                if cur is not None:
                    rows.append(cur)
                cur = {"epoch": int(m.group(1))}
                continue
            if cur is None:
                continue
            m = train_re.match(line)
            if m:
                cur.update({
                    "train_loss": float(m.group(1)),
                    "train_macro_f1": float(m.group(2)),
                    "train_ransom_P": float(m.group(3)), "train_ransom_R": float(m.group(4)),
                    "train_benign_P": float(m.group(5)), "train_benign_R": float(m.group(6)),
                    "train_malware_P": float(m.group(7)), "train_malware_R": float(m.group(8)),
                })
                continue
            m = val_re.match(line)
            if m:
                cur.update({
                    "val_loss": float(m.group(1)),
                    "val_macro_f1": float(m.group(2)),
                    "val_ransom_P": float(m.group(3)), "val_ransom_R": float(m.group(4)),
                    "val_benign_P": float(m.group(5)), "val_benign_R": float(m.group(6)),
                    "val_malware_P": float(m.group(7)), "val_malware_R": float(m.group(8)),
                })
                continue
            m = tvb_re.search(line)
            if m:
                cur.update({
                    "val_threat_R": float(m.group(2)),
                    "val_threat_F1": float(m.group(3)),
                    "val_benign_F1": float(m.group(6)),
                    "val_acc_2c": float(m.group(7)),
                    "val_threats_missed": int(m.group(8)),
                })
    if cur is not None:
        rows.append(cur)

    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return df


def fig9_training_curves() -> None:
    print("[fig9] Training curves parsed from results/train.log")

    log_path = PROJECT_ROOT / "results" / "train.log"
    if not log_path.exists():
        raise FileNotFoundError(f"Training log missing: {log_path}")

    df = _parse_train_log(log_path)
    if df.empty:
        raise RuntimeError("No epoch records parsed from training log")

    # Best epoch by val macro-F1 (with both Train/Val complete)
    complete = df.dropna(subset=["val_macro_f1", "train_macro_f1"])
    best_epoch = int(complete.loc[complete["val_macro_f1"].idxmax(), "epoch"])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))

    # ── (a) Loss curves
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], marker="o", color=C_TGNN, lw=1.8,
            markersize=5, markeredgecolor="black", markeredgewidth=0.4, label="Train")
    ax.plot(df["epoch"], df["val_loss"],   marker="s", color=C_RANSOM, lw=1.8,
            markersize=5, markeredgecolor="black", markeredgewidth=0.4, label="Validation")
    ax.axvline(best_epoch, color="#444444", linestyle=":", linewidth=1.0)
    ax.text(best_epoch, ax.get_ylim()[1] * 0.92,
            f" best epoch {best_epoch}", fontsize=8.5, color="#444444", va="top")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (focal)")
    ax.set_title("(a) Loss")
    ax.legend(loc="upper right")
    ax.set_xticks(df["epoch"][::max(1, len(df)//12)])

    # ── (b) Macro-F1
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_macro_f1"], marker="o", color=C_TGNN, lw=1.8,
            markersize=5, markeredgecolor="black", markeredgewidth=0.4, label="Train")
    ax.plot(df["epoch"], df["val_macro_f1"],   marker="s", color=C_RANSOM, lw=1.8,
            markersize=5, markeredgecolor="black", markeredgewidth=0.4, label="Validation")
    ax.axvline(best_epoch, color="#444444", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro-F1 (3-class)")
    ax.set_title("(b) Macro-F1")
    ax.set_ylim(min(df[["train_macro_f1", "val_macro_f1"]].min().min() - 0.05, 0.5), 1.02)
    ax.legend(loc="lower right")
    ax.set_xticks(df["epoch"][::max(1, len(df)//12)])

    # ── (c) Per-class validation recall
    ax = axes[1, 0]
    ax.plot(df["epoch"], df["val_ransom_R"],  marker="o", color=C_RANSOM, lw=1.8,
            markersize=5, markeredgecolor="black", markeredgewidth=0.4, label="Ransomware")
    ax.plot(df["epoch"], df["val_benign_R"],  marker="s", color=C_BENIGN, lw=1.8,
            markersize=5, markeredgecolor="black", markeredgewidth=0.4, label="Benign")
    ax.plot(df["epoch"], df["val_malware_R"], marker="D", color=C_MALWARE, lw=1.8,
            markersize=5, markeredgecolor="black", markeredgewidth=0.4, label="Other malware")
    ax.axvline(best_epoch, color="#444444", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Recall (validation)")
    ax.set_title("(c) Per-class validation recall")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncols=3, fontsize=8.5)
    ax.set_xticks(df["epoch"][::max(1, len(df)//12)])

    # ── (d) Threats missed + threat recall
    ax = axes[1, 1]
    ax2 = ax.twinx()
    if "val_threats_missed" in df.columns:
        ax.bar(df["epoch"], df["val_threats_missed"], color="#cccccc", edgecolor="black",
               linewidth=0.4, label="Threats missed (FN)", width=0.7, zorder=1)
    if "val_threat_R" in df.columns:
        ax2.plot(df["epoch"], df["val_threat_R"], marker="^", color=C_TGNN_DARK, lw=1.8,
                 markersize=6, markeredgecolor="black", markeredgewidth=0.4,
                 label="Threat recall (Ransom + Malware)", zorder=3)
    ax.axvline(best_epoch, color="#444444", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Threats missed (FN) on val")
    ax2.set_ylabel("Threat recall")
    ax2.set_ylim(0.90, 1.005)
    ax.set_title("(d) Validation threats missed and threat recall")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncols=2, fontsize=9, frameon=False)
    ax.set_xticks(df["epoch"][::max(1, len(df)//12)])
    ax.grid(False)
    ax2.grid(False)

    best_f1 = float(complete.loc[complete["val_macro_f1"].idxmax(), "val_macro_f1"])
    fig.suptitle(
        f"Training dynamics over {len(df)} epochs — best validation macro-F1 = {best_f1:.4f} "
        f"at epoch {best_epoch}; 5 918 train / 1 256 val graphs",
        fontsize=11.5, y=1.00,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    _savefig(fig, "fig9_training_curves")


# ═════════════════════════════════════════════════════════════════════════════
#  main
# ═════════════════════════════════════════════════════════════════════════════

FIGURES = [
    ("Figure 1", fig1_overall_performance),
    ("Figure 2", fig2_dataset_characterization),
    ("Figure 3", fig3_static_vs_temporal),
    ("Figure 4", fig4_robustness_dual_attack),
    ("Figure 5", fig5_semantic_substitution),
    ("Figure 6", fig6_component_attribution),
    ("Figure 7", fig7_branch_usage),
    ("Figure 8", fig8_zero_day_distribution_shift),
    ("Figure 9", fig9_training_curves),
]


def main(only: List[str] | None = None) -> None:
    print(f"Writing figures to: {PLOTS_DIR}\n")
    failures = []
    for name, fn in FIGURES:
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {name} failed: {type(e).__name__}: {e}")
            failures.append((name, e))
    print()
    if failures:
        print(f"⚠ {len(failures)} figure(s) failed:")
        for name, e in failures:
            print(f"   - {name}: {e}")
        sys.exit(1)
    print(f"✓ All {len(FIGURES)} figures generated under {PLOTS_DIR}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(only=args if args else None)
