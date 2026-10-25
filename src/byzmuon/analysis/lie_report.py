"""Figures for the 'A Little Is Enough' (LIE) attack alone: how damage scales with the attack strength z, when it
accrues, and what the attack does to the aggregate that the server orthogonalizes.

  lie_curves.*      test accuracy and test loss vs round, one panel per aggregator, one line per z (plus no attack)
  lie_dose.*        final accuracy vs z; accuracy gap vs round; rounds to reach a target accuracy vs z
  lie_aggregate.*   what the server sees: aggregate norm (relative to no attack) and effective rank vs round
  lie_shift.*       simulated per-coordinate shift of each aggregator as a function of z (why the formula z is tiny
                    and why the median saturates)
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from byzmuon.analysis.plots import AXIS, CAT, INK, INK2, MUTED, _label, _save, load_curves
from byzmuon.runner.results import load_results

NONE_STYLE = {"color": INK2, "linestyle": "--", "linewidth": 1.6}
Z_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark: weak -> strong attack (sequential, one hue)
AGG_ORDER = ["mean", "median", "trimmed_mean", "krum", "bulyan", "geomed", "centered_clip"]
AGG_LABEL = {"trimmed_mean": "trimmed mean", "centered_clip": "centered clip", "geomed": "geometric median"}


def lie_formula_z(n: int, f: int) -> float:
    """z of Baruch et al. 2019 for n clients and f attackers (the value the attack uses when z is not given)."""
    from statistics import NormalDist

    s = math.floor(n / 2 + 1) - f
    p = max(1e-6, min(1 - 1e-6, (n - f - s) / (n - f)))
    return NormalDist().inv_cdf(p)


def _lie_runs(df: pd.DataFrame) -> pd.DataFrame:
    """No-attack runs (z = 0) and LIE runs with their numeric z."""
    df = _label(df)
    n = int(df["cfg.n_clients"].iloc[0])
    f = int(df["cfg.num_byz"].iloc[0])
    is_lie = df["cfg.attack.name"] == "lie"
    is_none = df["cfg.attack.name"] == "none"
    sub = df[is_lie | is_none].copy()
    zcol = sub["cfg.attack.z"] if "cfg.attack.z" in sub.columns else pd.Series(np.nan, index=sub.index)
    sub["z"] = np.where(is_none[sub.index], 0.0, zcol.fillna(lie_formula_z(n, f)).astype(float))
    sub["z"] = sub["z"].round(3)
    sub["attacked"] = is_lie[sub.index].to_numpy()
    return sub


def _attack_zs(df: pd.DataFrame) -> list[float]:
    return sorted(df.loc[df["attacked"], "z"].unique())


def _all_zs(df: pd.DataFrame) -> list[float]:
    return [0.0] + _attack_zs(df) if (~df["attacked"]).any() else _attack_zs(df)


def _z_label(z: float) -> str:
    return "no attack" if z == 0 else f"LIE z={z:g}"


def _z_colors(attack_zs: list[float]) -> dict[float, str]:
    ramp = Z_RAMP if len(attack_zs) <= len(Z_RAMP) else Z_RAMP * (len(attack_zs) // len(Z_RAMP) + 1)
    return {z: ramp[len(ramp) - len(attack_zs) + i] for i, z in enumerate(attack_zs)}  # strongest = darkest


def _z_style(z: float, colors: dict[float, str]) -> dict:
    return dict(NONE_STYLE) if z == 0 else {"color": colors[z], "linewidth": 1.8}


def _aggs(df: pd.DataFrame) -> list[str]:
    present = set(df["agg"])
    return [a for a in AGG_ORDER if a in present] + sorted(present - set(AGG_ORDER))


def _agg_color(agg: str, aggs: list[str]) -> str:
    return CAT[aggs.index(agg) % len(CAT)]


def _agg_name(agg: str) -> str:
    return AGG_LABEL.get(agg, agg)


def _mean_curve(c: pd.DataFrame, col: str) -> pd.Series:
    return c.dropna(subset=[col]).groupby("round")[col].mean()


def _sel(curves: pd.DataFrame, agg: str, z: float) -> pd.DataFrame:
    return curves[(curves["agg"] == agg) & (curves["z"] == z)]


def _z_axis(ax, zs: list[float]) -> list[int]:
    """Equally spaced categorical positions for the z values (0.157 and 1 would collide on a linear axis)."""
    pos = list(range(len(zs)))
    ax.set_xticks(pos, ["none" if z == 0 else f"{z:g}" for z in zs])
    ax.set_xlabel("attack strength z")
    return pos


def fig_lie_curves(curves: pd.DataFrame, out: Path) -> list[Path]:
    aggs = _aggs(curves)
    zs = _all_zs(curves)
    colors = _z_colors(_attack_zs(curves))
    metrics = [("test_acc", "test accuracy")] + ([("test_loss", "test loss")] if "test_loss" in curves.columns else [])
    fig, axes = plt.subplots(
        len(metrics),
        len(aggs),
        figsize=(3.4 * len(aggs), 2.9 * len(metrics)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for j, agg in enumerate(aggs):
        for i, (col, name) in enumerate(metrics):
            ax = axes[i][j]
            for z in zs:
                c = _sel(curves, agg, z)
                if c.empty:
                    continue
                m = _mean_curve(c, col)
                ax.plot(m.index, m.values, label=_z_label(z), **_z_style(z, colors))
            if i == 0:
                ax.set_title(f"aggregator: {_agg_name(agg)}", color=INK)
                ax.set_ylim(0, 1)
            if i == len(metrics) - 1:
                ax.set_xlabel("round")
            if j == 0:
                ax.set_ylabel(name)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.subplots_adjust(bottom=0.14, hspace=0.12, wspace=0.08)
    fig.suptitle("FedMuon under the LIE attack: training curves for each attack strength z", color=INK, y=0.995)
    return _save(fig, out, "lie_curves")


def _rounds_to(c: pd.DataFrame, target: float) -> float:
    """First round at which the mean accuracy curve reaches `target`, linearly interpolated between evaluations."""
    m = _mean_curve(c, "test_acc")
    r, a = m.index.to_numpy(dtype=float), m.to_numpy()
    for i in range(1, len(a)):
        if a[i] >= target > a[i - 1]:
            return r[i - 1] + (target - a[i - 1]) / (a[i] - a[i - 1]) * (r[i] - r[i - 1])
    return 0.0 if len(a) and a[0] >= target else np.nan


def _smooth(s: pd.Series, k: int = 3) -> pd.Series:
    return s.rolling(k, center=True, min_periods=1).mean()


def fig_lie_dose(df: pd.DataFrame, curves: pd.DataFrame, out: Path, target: float = 0.7) -> list[Path]:
    aggs = _aggs(df)
    zs = _all_zs(df)
    attack_zs = _attack_zs(df)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")

    # (a) final accuracy vs z, labelled with the drop in percentage points
    ax = axes[0]
    pos = _z_axis(ax, zs)
    for agg in aggs:
        g = df[df["agg"] == agg].groupby("z")["final_test_acc"].agg(["mean", "std"]).reindex(zs)
        ax.errorbar(
            pos,
            g["mean"],
            yerr=g["std"].fillna(0) if g["std"].notna().any() else None,
            marker="o",
            markersize=5,
            color=_agg_color(agg, aggs),
            label=_agg_name(agg),
            capsize=2,
        )
        base = g["mean"].get(0.0, np.nan)
        for p, (z, v) in zip(pos, g["mean"].items(), strict=True):
            if z != 0 and not (np.isnan(base) or np.isnan(v)):
                ax.annotate(
                    f"{100 * (base - v):+.1f}",
                    (p, v),
                    textcoords="offset points",
                    xytext=(0, -11 - 8 * aggs.index(agg)),
                    ha="center",
                    fontsize=7,
                    color=INK2,
                )
    ax.set_ylabel("final test accuracy")
    ax.set_title("final accuracy (labels: loss in pp)", color=INK)
    ax.legend(frameon=False, loc="lower left")

    # (b) accuracy lost vs round, strongest z solid, weaker z dashed/dotted (3-point moving average)
    ax = axes[1]
    styles = ["-", "--", ":", "-."]
    for agg in aggs:
        base = _mean_curve(_sel(curves, agg, 0.0), "test_acc")
        for k, z in enumerate(sorted(attack_zs, reverse=True)):
            c = _sel(curves, agg, z)
            if c.empty or base.empty:
                continue
            gap = _smooth((base - _mean_curve(c, "test_acc")).dropna())
            ax.plot(
                gap.index,
                100 * gap.values,
                styles[k % len(styles)],
                color=_agg_color(agg, aggs),
                linewidth=1.6,
                label=f"{_agg_name(agg)}, z={z:g}",
            )
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("round")
    ax.set_ylabel("accuracy lost to LIE (pp)")
    ax.set_title("when the damage accrues", color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=len(aggs), loc="upper center", bbox_to_anchor=(0.5, -0.22))

    # (c) rounds to reach the target accuracy vs z
    ax = axes[2]
    pos = _z_axis(ax, zs)
    never = []
    for agg in aggs:
        rt = [_rounds_to(_sel(curves, agg, z), target) for z in zs]
        ax.plot(pos, rt, marker="o", markersize=5, color=_agg_color(agg, aggs), label=_agg_name(agg))
        never += [(p, agg) for p, r in zip(pos, rt, strict=True) if np.isnan(r)]
    for p, agg in never:
        ax.annotate(f"{_agg_name(agg)}: never", (p, ax.get_ylim()[1]), ha="center", fontsize=7, color=MUTED)
    ax.set_ylabel(f"rounds to reach {100 * target:.0f}% accuracy")
    ax.set_title("slowdown", color=INK)
    fig.suptitle("Dose-response of FedMuon to the LIE attack", color=INK)
    return _save(fig, out, "lie_dose")


def simulate_shift(n: int, f: int, zs, aggs, trials: int = 20000, seed: int = 0) -> dict[str, list[float]]:
    """Per-coordinate shift of the aggregate, in honest-std units, when f of n clients send mean − z·std.
    Honest coordinates are i.i.d. standard normal; trimmed mean trims f on each side."""
    rng = np.random.default_rng(seed)
    honest = rng.standard_normal((trials, n - f))
    mu, sd = honest.mean(1, keepdims=True), honest.std(1, keepdims=True)
    clean = {
        "mean": honest.mean(1),
        "median": np.median(honest, 1),
        "trimmed_mean": np.sort(honest, 1)[:, f : n - 2 * f].mean(1) if n - 3 * f > 0 else honest.mean(1),
    }
    out = {a: [] for a in aggs}
    for z in zs:
        allv = np.concatenate([honest, np.repeat(mu - z * sd, f, axis=1)], axis=1)
        attacked = {
            "mean": allv.mean(1),
            "median": np.median(allv, 1),
            "trimmed_mean": np.sort(allv, 1)[:, f : n - f].mean(1),
        }
        for a in aggs:
            out[a].append(float(((clean[a] - attacked[a]) / sd[:, 0]).mean()) if a in attacked else np.nan)
    return out


def fig_lie_shift(df: pd.DataFrame, out: Path) -> list[Path]:
    aggs = [a for a in _aggs(df) if a in ("mean", "median", "trimmed_mean")]
    attack_zs = _attack_zs(df)
    colors = _z_colors(attack_zs)
    n, f = int(df["cfg.n_clients"].iloc[0]), int(df["cfg.num_byz"].iloc[0])
    fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
    zgrid = np.linspace(0, max(2.5, max(attack_zs) * 1.1), 80)
    sim = simulate_shift(n, f, zgrid, aggs)
    for agg in aggs:
        ax.plot(zgrid, sim[agg], color=_agg_color(agg, _aggs(df)), linewidth=1.8, label=_agg_name(agg))
    for z in attack_zs:
        ax.axvline(z, color=colors[z], linewidth=0.9, linestyle=":")
        ax.annotate(f"z={z:g}", (z, 0), xytext=(3, 2), textcoords="offset points", fontsize=7, color=colors[z])
    zf = lie_formula_z(n, f)
    ax.annotate(
        f"formula z = {zf:.2f}\n(shifts the mean by {100 * f / n * zf:.0f}% of one std)",
        (zf, sim["mean"][int(np.searchsorted(zgrid, zf))] if "mean" in sim else 0),
        xytext=(1.25, 0.05),
        textcoords="data",
        fontsize=7,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )
    ax.set_xlabel("attack strength z")
    ax.set_ylabel("shift of the aggregate\n(per coordinate, in honest-std units)")
    ax.set_title(f"How far LIE moves each aggregator, n={n}, f={f} (simulated)", color=INK)
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, out, "lie_shift")


def fig_lie_aggregate(df: pd.DataFrame, curves: pd.DataFrame, out: Path) -> list[Path]:
    aggs = _aggs(df)
    zs = _all_zs(df)
    colors = _z_colors(_attack_zs(df))
    erank_cols = [c for c in curves.columns if c.startswith("spec/") and c.endswith("/erank")]
    rows = (["fro"] if "agg_fro" in curves.columns else []) + (["erank"] if erank_cols else [])
    if not rows:
        return []
    if erank_cols:
        curves = curves.assign(erank=curves[erank_cols].mean(axis=1))
    fig, axes = plt.subplots(
        len(rows), len(aggs), figsize=(3.6 * len(aggs), 2.9 * len(rows)), sharex=True, sharey="row", squeeze=False
    )
    for j, agg in enumerate(aggs):
        for i, row in enumerate(rows):
            ax = axes[i][j]
            if row == "fro":
                base = _mean_curve(_sel(curves, agg, 0.0), "agg_fro")
                for z in zs:
                    if z == 0:
                        continue
                    ratio = _smooth((_mean_curve(_sel(curves, agg, z), "agg_fro") / base).dropna(), 5)
                    ax.plot(ratio.index, ratio.values, label=_z_label(z), **_z_style(z, colors))
                ax.axhline(1, **NONE_STYLE)
                if j == 0:
                    ax.set_ylabel("‖aggregate‖ relative to no attack")
                if i == 0:
                    ax.set_title(f"aggregator: {_agg_name(agg)}", color=INK)
            else:
                for z in zs:
                    m = _mean_curve(_sel(curves, agg, z), "erank")
                    ax.plot(m.index, m.values, label=_z_label(z), **_z_style(z, colors))
                if j == 0:
                    ax.set_ylabel("effective rank of the aggregate")
            if i == len(rows) - 1:
                ax.set_xlabel("round")
    handles, labels = axes[-1][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.subplots_adjust(bottom=0.14, hspace=0.12, wspace=0.08)
    fig.suptitle(
        "What the Muon server sees under LIE: a larger aggregate (top) with fewer significant directions (bottom)",
        color=INK,
        y=0.995,
    )
    return _save(fig, out, "lie_aggregate")


def make_lie_figures(results_dir: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    results_dir = Path(results_dir)
    out = Path(out_dir) if out_dir else results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df = _lie_runs(load_results(results_dir))
    if not df["attacked"].any():
        raise ValueError(f"no LIE runs in {results_dir}")
    curves = load_curves(results_dir, df).merge(df[["hash", "z", "attacked"]], on="hash")
    return (
        fig_lie_curves(curves, out)
        + fig_lie_dose(df, curves, out)
        + fig_lie_aggregate(df, curves, out)
        + fig_lie_shift(df, out)
    )
