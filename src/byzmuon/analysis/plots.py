"""Paper-style figures and summary tables from a results directory.

Figures (PNG + PDF, light surface, one y-axis each):
  curves.*        test accuracy vs round, one panel per aggregator; color = attack, line style = server optimizer
  drop.*          accuracy drop (no attack − attack) per aggregator, bars by optimizer, error bars over seeds
  amplification.* orthogonalization gain vs round per layer (log scale), reference line at 1
  spectrum.*      effective rank of the honest mean per layer vs round
Tables: summary.csv and summary.md with final accuracy mean ± std and the drop per cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from byzmuon.runner.results import load_results  # noqa: E402

# Categorical palette (validated, fixed order: identity never cycles). Attacks and optimizers each get a fixed slot.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ATTACK_ORDER = [
    "none",
    "lie",
    "subgap",
    "spike",
    "minmax",
    "minsum",
    "signflip",
    "ipm",
    "gaussian",
    "labelflip",
    "intermittent",
]
OPT_ORDER = ["muon", "normsgd", "momsgd", "sgd", "adam"]
OPT_STYLE = {"muon": "-", "normsgd": "--", "momsgd": "--", "sgd": "--", "adam": ":"}
OPT_LABEL = {
    "muon": "FedMuon",
    "normsgd": "Momentum SGD (normalized)",
    "momsgd": "Momentum SGD",
    "sgd": "FedAvg",
    "adam": "FedAdam",
}
STYLE_WORD = {"-": "solid", "--": "dashed", ":": "dotted"}
MODEL_LABEL = {"resnet18": "ResNet-18", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cnn": "CNN", "mlp": "MLP"}
AGG_LABEL = {"trimmed_mean": "trimmed mean", "centered_clip": "centered clip", "geomed": "geometric median"}
ATTACK_LABEL = {"none": "no attack", "lie": "LIE", "minmax": "Min-Max", "minsum": "Min-Sum", "ipm": "IPM"}


def _attack_name(attack: str) -> str:
    """'lie z=0.25' -> 'LIE z=0.25', 'minmax' -> 'Min-Max'."""
    head, _, rest = attack.partition(" ")
    return (ATTACK_LABEL.get(head, head) + (" " + rest if rest else "")).strip()


BLUE_RAMP = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
    }
)


def _attack_order(labels) -> list[str]:
    """Attack labels sorted by family (ATTACK_ORDER) then variant, so colors are assigned in a fixed order."""
    fam = {n: i for i, n in enumerate(ATTACK_ORDER)}
    return sorted(set(labels), key=lambda a: (fam.get(a.split()[0], len(fam)), a))


def _color_for(name: str, order: list[str]) -> str:
    return CAT[order.index(name) % len(CAT)] if name in order else CAT[-1]


def _attack_labels(df: pd.DataFrame) -> pd.Series:
    """'name' plus any attack kwargs that differ between runs of the same name, e.g. 'lie z=1.0'."""
    cols = [
        c for c in df.columns if c.startswith("cfg.attack.") and c not in ("cfg.attack.name", "cfg.attack.knowledge")
    ]
    out = df["cfg.attack.name"].astype(str).copy()
    for name, grp in df.groupby("cfg.attack.name"):
        varying = [c for c in cols if grp[c].astype(str).nunique() > 1]
        if varying:
            tags = [
                " ".join(f"{c.rsplit('.', 1)[1]}={row[c] if pd.notna(row[c]) else 'auto'}" for c in varying)
                for _, row in grp.iterrows()
            ]
            out.loc[grp.index] = [f"{name} {t}" for t in tags]
    return out


def _label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["attack"] = _attack_labels(df)
    df["opt"] = df["cfg.server_opt.name"]
    if "cfg.client_opt.local_momentum" in df.columns:  # server SGD over local momentum buffers = momentum SGD
        mom = df["cfg.client_opt.local_momentum"].fillna(0).astype(float) > 0
        df.loc[mom & (df["opt"] == "sgd"), "opt"] = "momsgd"
    df["agg"] = df["cfg.aggregator.base.name"]
    df["seed"] = df["cfg.seed"]
    df["model"] = df["cfg.model.name"] if "cfg.model.name" in df.columns else "model"
    return df


def load_curves(results_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        p = results_dir / r["hash"] / "metrics.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            m.update(
                {
                    "hash": r["hash"],
                    "attack": r["attack"],
                    "opt": r["opt"],
                    "agg": r["agg"],
                    "seed": r["seed"],
                    "model": r["model"],
                }
            )
            rows.append(m)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        df = df.assign(model="model")
    g = df.groupby(["model", "opt", "agg", "attack"])["final_test_acc"].agg(["mean", "std", "count"]).reset_index()
    base = g[g["attack"] == "none"].set_index(["model", "opt", "agg"])["mean"]
    g["drop"] = [
        base.get((mo, o, a), np.nan) - m for mo, o, a, m in zip(g["model"], g["opt"], g["agg"], g["mean"], strict=True)
    ]
    g["std"] = g["std"].fillna(0.0)
    return g.sort_values(["model", "agg", "attack", "opt"]).reset_index(drop=True)


def _save(fig, out: Path, name: str) -> list[Path]:
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def fig_curves(curves: pd.DataFrame, out: Path) -> list[Path]:
    if "test_acc" not in curves.columns:
        return []
    c = curves.dropna(subset=["test_acc"])
    paths = []
    for model in sorted(c["model"].unique()):
        paths += _fig_curves_one(c[c["model"] == model], model, out)
    return paths


def _fig_curves_one(c: pd.DataFrame, model: str, out: Path) -> list[Path]:
    aggs = sorted(c["agg"].unique())
    fig, axes = plt.subplots(1, len(aggs), figsize=(3.4 * len(aggs), 3.2), sharey=True, squeeze=False)
    order = _attack_order(c["attack"])
    for ax, agg in zip(axes[0], aggs, strict=True):
        sub = c[c["agg"] == agg]
        for (attack, opt), grp in sub.groupby(["attack", "opt"]):
            per_round = grp.groupby("round")["test_acc"]
            mean, lo, hi = per_round.mean(), per_round.min(), per_round.max()
            color = _color_for(attack, order)
            ax.plot(
                mean.index,
                mean.values,
                OPT_STYLE.get(opt, "-"),
                color=color,
                label=f"{OPT_LABEL.get(opt, opt)}, {attack}",
            )
            if grp["seed"].nunique() > 1:
                ax.fill_between(mean.index, lo.values, hi.values, color=color, alpha=0.12, linewidth=0)
        ax.set_title(f"aggregator: {AGG_LABEL.get(agg, agg)}", color=INK)
        ax.set_xlabel("round")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0][0].set_ylabel("test accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    ncol = min(4, len(labels))
    nrow = -(-len(labels) // ncol)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.subplots_adjust(bottom=0.16 + 0.03 * nrow)
    opts = [o for o in OPT_ORDER if o in set(c["opt"])]
    styles = ", ".join(f"{STYLE_WORD.get(OPT_STYLE.get(o, '-'), '')}: {OPT_LABEL.get(o, o)}" for o in opts)
    band = "; band: seed min–max" if c["seed"].nunique() > 1 else ""
    fig.suptitle(f"Test accuracy over rounds, {MODEL_LABEL.get(model, model)} ({styles}{band})", color=INK, y=1.02)
    return _save(fig, out, f"curves_{model}")


def fig_drop(summary: pd.DataFrame, out: Path) -> list[Path]:
    s = summary[summary["attack"] != "none"].dropna(subset=["drop"])
    if s.empty:
        return []
    if s["model"].nunique() > 1:  # one drop figure per model
        return [p for m in sorted(s["model"].unique()) for p in _fig_drop_one(s[s["model"] == m], out, f"drop_{m}")]
    return _fig_drop_one(s, out, "drop")


def _fig_drop_one(s: pd.DataFrame, out: Path, name: str) -> list[Path]:
    attacks = _attack_order(s["attack"])
    opts = [o for o in OPT_ORDER if o in set(s["opt"])]
    aggs = sorted(s["agg"].unique())
    ncol = min(4, len(attacks))
    nrow = -(-len(attacks) // ncol)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(1.0 + (1.2 + 0.7 * len(aggs)) * ncol, 3.0 * nrow),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.flat[len(attacks) :]:
        ax.set_visible(False)
    width = 0.8 / max(1, len(opts))
    for ax, attack in zip(axes.flat, attacks, strict=False):
        sub = s[s["attack"] == attack]
        x = np.arange(len(aggs))
        for i, opt in enumerate(opts):
            vals = [sub[(sub["agg"] == a) & (sub["opt"] == opt)]["drop"].mean() for a in aggs]
            errs = [sub[(sub["agg"] == a) & (sub["opt"] == opt)]["std"].mean() for a in aggs]
            pos = x + (i - (len(opts) - 1) / 2) * width
            ax.bar(
                pos,
                vals,
                width * 0.9,
                color=_color_for(opt, OPT_ORDER),
                label=OPT_LABEL.get(opt, opt),
                yerr=errs if any(e > 0 for e in errs) else None,
                ecolor=INK2,
                capsize=2,
                linewidth=0,
            )
            for p_, v in zip(pos, vals, strict=True):
                if not np.isnan(v):
                    ax.text(
                        p_,
                        v + 0.005 * np.sign(v or 1),
                        f"{100 * v:.1f}",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=7,
                        color=INK2,
                    )
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.set_xticks(x, [AGG_LABEL.get(a, a) for a in aggs], rotation=20)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(f"attack: {attack}", color=INK)
    axes[0][0].set_ylabel("accuracy lost vs. no attack")
    axes[0][0].legend(loc="upper left")
    errs_note = "; error bars: std over seeds" if s["std"].fillna(0).gt(0).any() else ""
    fig.suptitle(f"Accuracy lost to the attack (percentage points{errs_note})", color=INK)
    return _save(fig, out, name)


def fig_attack_compare(summary: pd.DataFrame, out: Path) -> list[Path]:
    """Final accuracy under every attack, one panel per aggregator, grouped bars per optimizer: the one-figure
    answer to 'does FedMuon hold up better than momentum SGD?'. Only drawn when there are >= 2 optimizers."""
    if summary.empty or summary["opt"].nunique() < 2:
        return []
    paths = []
    for model in sorted(summary["model"].unique()):
        s = summary[summary["model"] == model]
        name = "attack_compare" if summary["model"].nunique() == 1 else f"attack_compare_{model}"
        paths += _fig_attack_compare_one(s, model, out, name)
    return paths


def _fig_attack_compare_one(s: pd.DataFrame, model: str, out: Path, name: str) -> list[Path]:
    attacks = _attack_order(s["attack"])
    opts = [o for o in OPT_ORDER if o in set(s["opt"])] + sorted(set(s["opt"]) - set(OPT_ORDER))
    aggs = sorted(s["agg"].unique())
    fig, axes = plt.subplots(
        len(aggs), 1, figsize=(1.5 + 0.95 * len(attacks) * len(opts), 2.9 * len(aggs)), sharex=True, squeeze=False
    )
    width = 0.8 / len(opts)
    x = np.arange(len(attacks))
    for ax, agg in zip(axes[:, 0], aggs, strict=True):
        sub = s[s["agg"] == agg]
        for i, opt in enumerate(opts):
            rows = [sub[(sub["attack"] == a) & (sub["opt"] == opt)] for a in attacks]
            vals = [float(r["mean"].iloc[0]) if len(r) else np.nan for r in rows]
            errs = [float(r["std"].iloc[0]) if len(r) else 0.0 for r in rows]
            pos = x + (i - (len(opts) - 1) / 2) * width
            ax.bar(
                pos,
                vals,
                width * 0.9,
                color=_color_for(opt, OPT_ORDER),
                label=OPT_LABEL.get(opt, opt),
                yerr=errs if any(e > 0 for e in errs) else None,
                ecolor=INK2,
                capsize=2,
                linewidth=0,
            )
            for p_, v in zip(pos, vals, strict=True):
                if not np.isnan(v):
                    ax.text(p_, v + 0.01, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("final accuracy")
        ax.set_title(f"aggregator: {AGG_LABEL.get(agg, agg)}", color=INK, loc="left")
    axes[-1, 0].set_xticks(x, [_attack_name(a) for a in attacks], rotation=20, ha="right")
    axes[0, 0].legend(loc="upper right", ncol=len(opts))
    fig.suptitle(f"Final test accuracy under each attack, {MODEL_LABEL.get(model, model)}", color=INK)
    fig.tight_layout()
    return _save(fig, out, name)


def _layer_cols(curves: pd.DataFrame, prefix: str, suffix: str) -> list[tuple[str, str]]:
    cols = [c for c in curves.columns if c.startswith(prefix) and c.endswith(suffix) and "/mean/" not in c]
    return [(c, c[len(prefix) : -len(suffix)].strip("/")) for c in cols]


def _clean_rows(curves: pd.DataFrame, model: str, needed: str | list[str]) -> pd.DataFrame:
    """Rows of the clean (no-attack) runs of one model that have at least one of the `needed` columns."""
    sub = curves[curves["model"] == model]
    if (sub["attack"] == "none").any():
        sub = sub[sub["attack"] == "none"]
    needed = [needed] if isinstance(needed, str) else [c for c in needed if c in sub.columns]
    return sub.dropna(subset=needed, how="all") if needed else sub.iloc[0:0]


def _layer_lines(ax, sub: pd.DataFrame, layer_cols: list[tuple[str, str]], linewidth: float = 1.1) -> None:
    """One line per layer, blue ramp ordered by layer name (roughly depth); legend carries the names."""
    present = [(c, layer) for c, layer in layer_cols if sub[c].notna().any()]
    ramp = [BLUE_RAMP[int(i * (len(BLUE_RAMP) - 1) / max(1, len(present) - 1))] for i in range(len(present))]
    for color, (c, layer) in zip(ramp, present, strict=True):
        per = sub.groupby("round")[c].mean()
        ax.plot(per.index, per.values, color=color, linewidth=linewidth, label=layer.replace(".weight", ""))


def fig_amplification(curves: pd.DataFrame, out: Path) -> list[Path]:
    layer_cols = _layer_cols(curves, "amp/", "/tail/0.01")
    if not layer_cols or "amp/mean/tail/0.01" not in curves.columns:
        return []
    models = [m for m in sorted(curves["model"].unique()) if not _clean_rows(curves, m, "amp/mean/tail/0.01").empty]
    if not models:
        return []
    fig, axes = plt.subplots(
        1, len(models), figsize=(7.2 * len(models), 3.4), squeeze=False, sharey=True, layout="constrained"
    )
    for ax, model in zip(axes[0], models, strict=True):
        sub = _clean_rows(curves, model, "amp/mean/tail/0.01")
        _layer_lines(ax, sub, layer_cols)
        mean = sub.groupby("round")["amp/mean/tail/0.01"].mean()
        ax.plot(
            mean.index,
            mean.values,
            color=CAT[1],
            linewidth=2.4,
            label="mean over layers, trailing-subspace perturbation",
        )
        if "amp/mean/random/0.01" in sub.columns:
            rnd = sub.groupby("round")["amp/mean/random/0.01"].mean()
            ax.plot(
                rnd.index,
                rnd.values,
                color=CAT[1],
                linewidth=1.5,
                linestyle="--",
                label="mean over layers, random perturbation",
            )
        ax.axhline(1.0, color=AXIS, linewidth=0.9)
        ax.text(mean.index[0], 1.07, "linear aggregator: gain 1", fontsize=7, color=MUTED)
        ax.set_yscale("log")
        ax.set_xlabel("round")
        ax.set_title(f"model: {model}", color=INK)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5, title="layer", title_fontsize=7)
    axes[0][0].set_ylabel("amplification gain (ε = 1%)")
    fig.suptitle(
        "Orthogonalization gain on the honest mean: relative change of the update per unit relative perturbation",
        color=INK,
        y=1.02,
        fontsize=9.5,
    )
    return _save(fig, out, "amplification")


def fig_spectrum(curves: pd.DataFrame, out: Path) -> list[Path]:
    layer_cols = _layer_cols(curves, "hspec/", "/erank")
    if not layer_cols:
        return []
    erank_cols = [c for c, _ in layer_cols]
    models = [m for m in sorted(curves["model"].unique()) if not _clean_rows(curves, m, erank_cols).empty]
    if not models:
        return []
    fig, axes = plt.subplots(len(models), 2, figsize=(9.2, 3.1 * len(models)), squeeze=False, layout="constrained")
    for row, model in zip(axes, models, strict=True):
        sub = _clean_rows(curves, model, erank_cols)
        _layer_lines(row[0], sub, layer_cols, linewidth=1.3)
        tail_cols = [
            (f"hspec/{layer}/tail_frac", layer) for _, layer in layer_cols if f"hspec/{layer}/tail_frac" in sub.columns
        ]
        _layer_lines(row[1], sub, tail_cols, linewidth=1.3)
        row[0].set_ylabel("effective rank")
        row[1].set_ylabel("near-zero direction fraction")
        row[1].set_ylim(0, 1)
        row[0].set_title(f"{model}: effective rank of the honest mean", color=INK)
        row[1].set_title(f"{model}: share of singular values below 1% of the largest", color=INK)
        row[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5, title="layer", title_fontsize=7)
        for ax in row:
            ax.set_xlabel("round")
    fig.suptitle("Spectrum of the honest aggregate per layer (darker = deeper layer)", color=INK)
    return _save(fig, out, "spectrum")


def write_summary(summary: pd.DataFrame, out: Path) -> list[Path]:
    csv = out / "summary.csv"
    summary.to_csv(csv, index=False)
    lines = [
        "| model | aggregator | attack | optimizer | final acc (mean ± std) | seeds | drop vs none |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        drop = "" if np.isnan(r["drop"]) else f"{100 * r['drop']:+.1f} pp"
        lines.append(
            f"| {r['model']} | {r['agg']} | {r['attack']} | {OPT_LABEL.get(r['opt'], r['opt'])} | "
            f"{100 * r['mean']:.1f} ± {100 * r['std']:.1f} | {int(r['count'])} | {drop} |"
        )
    md = out / "summary.md"
    md.write_text("\n".join(lines) + "\n")
    return [csv, md]


def make_figures(results_dir: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    results_dir = Path(results_dir)
    out = Path(out_dir) if out_dir else results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = load_results(results_dir)
    if df.empty:
        raise SystemExit(f"no finished runs under {results_dir}")
    df = _label(df)
    curves = load_curves(results_dir, df)
    summary = summarize(df)
    written = write_summary(summary, out)
    written += fig_curves(curves, out)
    written += fig_drop(summary, out)
    written += fig_attack_compare(summary, out)
    written += fig_amplification(curves, out)
    written += fig_spectrum(curves, out)
    return written
