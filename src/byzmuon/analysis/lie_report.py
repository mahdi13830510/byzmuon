"""Figures for the 'A Little Is Enough' (LIE) attack alone. Each figure makes one point and says it in its title.

lie_final.*       final accuracy per aggregator, one bar per attack strength z, loss in pp written on the bar
lie_timeline.*    accuracy lost to LIE vs round, one panel per aggregator (when the damage appears)
lie_mechanism.*   what the Muon server sees: size of the aggregate (which Muon normalizes away) and its effective
                  rank (which it does not), per aggregator and z
lie_geometry.*    one coordinate: where LIE places the f attackers relative to the n−f honest values, for each z
lie_shift.*       simulated per-coordinate shift of each aggregator as a function of z
lie_curves.*      test accuracy vs round with direct labels (supporting figure)
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import PercentFormatter

from byzmuon.analysis.plots import (
    AGG_LABEL,
    AXIS,
    CAT,
    GRID,
    INK,
    INK2,
    MODEL_LABEL,
    OPT_LABEL,
    OPT_ORDER,
    _label,
    _save,
    load_curves,
)
from byzmuon.runner.results import load_results

NONE_COLOR = "#9a9890"
Z_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark: weak -> strong attack (sequential, one hue)
AGG_ORDER = ["mean", "median", "trimmed_mean", "krum", "bulyan", "geomed", "centered_clip"]
THREE = ["mean", "median", "trimmed_mean"]
RC = {"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11, "legend.fontsize": 10, "figure.titlesize": 13}


def lie_formula_z(n: int, f: int) -> float:
    """z of Baruch et al. 2019 for n clients and f attackers (the value the attack uses when z is not given)."""
    from statistics import NormalDist

    s = math.floor(n / 2 + 1) - f
    p = max(1e-6, min(1 - 1e-6, (n - f - s) / (n - f)))
    return NormalDist().inv_cdf(p)


# ----------------------------------------------------------------------------------------------- data helpers


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


def _z_label(z: float, formula: float | None = None) -> str:
    if z == 0:
        return "no attack"
    if formula is not None and abs(z - formula) < 1e-3:
        return f"LIE z={z:g} (paper's formula)"
    return f"LIE z={z:g}"


def _z_colors(attack_zs: list[float]) -> dict[float, str]:
    """One distinct step of a light-to-dark blue ramp per attack strength (strongest = darkest), any count."""
    cmap = LinearSegmentedColormap.from_list("z", Z_RAMP)
    n = len(attack_zs)
    stops = np.linspace(1.0 - 0.3 * (n - 1), 1.0, n) if n <= 3 else np.linspace(0.0, 1.0, n)
    colors = {z: to_hex(cmap(t)) for z, t in zip(attack_zs, stops, strict=True)}
    colors[0.0] = NONE_COLOR
    return colors


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


def _smooth(s: pd.Series, k: int = 3) -> pd.Series:
    return s.rolling(k, center=True, min_periods=1).mean()


def _nf(df: pd.DataFrame) -> tuple[int, int]:
    return int(df["cfg.n_clients"].iloc[0]), int(df["cfg.num_byz"].iloc[0])


def _rounds(df: pd.DataFrame) -> int:
    return int(df["cfg.rounds"].iloc[0]) if "cfg.rounds" in df.columns else 0


def _setup(df: pd.DataFrame) -> str:
    n, f = _nf(df)
    model = str(df["cfg.model.name"].iloc[0]) if "cfg.model.name" in df.columns else "model"
    data = str(df["cfg.data.name"].iloc[0]) if "cfg.data.name" in df.columns else "data"
    setup = f"{MODEL_LABEL.get(model, model)} on {MODEL_LABEL.get(data, data)}, {f} of {n} clients Byzantine"
    if df["opt"].nunique() == 1:
        return f"{OPT_LABEL.get(str(df['opt'].iloc[0]), str(df['opt'].iloc[0]))}, {setup}"
    return setup


def _trimmed(vals: np.ndarray, f: int) -> float:
    k = min(f, (len(vals) - 1) // 2)
    return float(np.sort(vals)[k : len(vals) - k].mean())


# ----------------------------------------------------------------------------------------------- figures


def fig_lie_final(df: pd.DataFrame, out: Path) -> list[Path]:
    aggs, zs = _aggs(df), _all_zs(df)
    colors = _z_colors(_attack_zs(df))
    zf = lie_formula_z(*_nf(df))
    g = df.groupby(["agg", "z"])["final_test_acc"].agg(["mean", "std", "count"])
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(2.4 * len(aggs) + 3, 4.6), layout="constrained")
        width = 0.8 / len(zs)
        x = np.arange(len(aggs))
        for i, z in enumerate(zs):
            pos = x + (i - (len(zs) - 1) / 2) * width
            vals = [g["mean"].get((a, z), np.nan) for a in aggs]
            errs = [g["std"].get((a, z), np.nan) for a in aggs]
            ax.bar(
                pos,
                vals,
                width * 0.92,
                color=colors[z],
                label=_z_label(z, zf),
                yerr=errs if any(e > 0 for e in errs if not np.isnan(e)) else None,
                ecolor=INK2,
                capsize=2,
                linewidth=0,
            )
            for p, a, v in zip(pos, aggs, vals, strict=True):
                if np.isnan(v):
                    continue
                ax.text(p, v + 0.004, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
                base = g["mean"].get((a, 0.0), np.nan)
                if z != 0 and not np.isnan(base):
                    delta = 100 * (v - base)
                    ax.text(
                        p,
                        v - 0.012,
                        f"{delta:+.1f}" if abs(delta) >= 2 else "±0",
                        ha="center",
                        va="top",
                        fontsize=8.5,
                        fontweight="bold" if abs(delta) >= 2 else "normal",
                        color="white",
                    )
        lo = max(0.0, g["mean"].min() - 0.04)
        ax.set_ylim(lo, min(1.0, g["mean"].max() + 0.075))
        ax.set_xticks(x, [_agg_name(a) for a in aggs])
        ax.set_xlabel("aggregator at the server")
        ax.set_ylabel(f"test accuracy after {_rounds(df)} rounds")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.legend(frameon=False, loc="upper center", ncol=len(zs))
        ax.set_title(
            f"Final test accuracy under LIE  ({_setup(df)}; white: change vs. no attack, pp)",
            color=INK,
            loc="left",
        )
    return _save(fig, out, "lie_final")


def fig_lie_timeline(df: pd.DataFrame, curves: pd.DataFrame, out: Path) -> list[Path]:
    aggs, zs = _aggs(df), _attack_zs(df)
    colors = _z_colors(zs)
    zf = lie_formula_z(*_nf(df))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(aggs), figsize=(3.9 * len(aggs), 4.0), sharey=True, squeeze=False)
        for ax, agg in zip(axes[0], aggs, strict=True):
            base = _mean_curve(_sel(curves, agg, 0.0), "test_acc")
            ends = []
            for z in zs:
                c = _sel(curves, agg, z)
                if c.empty or base.empty:
                    continue
                gap = 100 * _smooth((base - _mean_curve(c, "test_acc")).dropna())
                ax.plot(gap.index, gap.values, color=colors[z], linewidth=2.0, label=_z_label(z, zf))
                ax.fill_between(gap.index, 0, gap.values, where=gap.values > 0, color=colors[z], alpha=0.10, lw=0)
                ends.append((gap.values[-1], gap.index[-1], colors[z]))
            ends.sort()
            last_y = -1e9
            for v, r, col in ends:  # direct labels, nudged apart
                y = max(v, last_y + 0.9)
                ax.text(r + 2, y, f"{v:+.1f}", color=col, fontsize=9, va="center")
                last_y = y
            ax.axhline(0, color=AXIS, linewidth=0.9)
            ax.set_title(f"aggregator: {_agg_name(agg)}", color=INK)
            ax.set_xlabel("round")
            if len(base):
                ax.set_xlim(0, base.index.max() * 1.12)
        axes[0][0].set_ylabel("accuracy loss vs. no attack (pp)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.02), frameon=False)
        fig.subplots_adjust(bottom=0.24, wspace=0.08, top=0.80, left=0.08, right=0.98)
        fig.suptitle(f"Accuracy loss under LIE over training  ({_setup(df)})", color=INK, x=0.01, ha="left")
    return _save(fig, out, "lie_timeline")


def fig_lie_mechanism(df: pd.DataFrame, curves: pd.DataFrame, out: Path, window: int = 25) -> list[Path]:
    aggs, zs = _aggs(df), _all_zs(df)
    colors = _z_colors(_attack_zs(df))
    zf = lie_formula_z(*_nf(df))
    erank_cols = [c for c in curves.columns if c.startswith("spec/") and c.endswith("/erank")]
    have_fro = "agg_fro" in curves.columns
    if not (have_fro or erank_cols):
        return []
    if erank_cols:
        curves = curves.assign(erank=curves[erank_cols].mean(axis=1))
    last_round = curves["round"].max()

    def late(agg: str, z: float, col: str) -> float:
        c = _sel(curves, agg, z).dropna(subset=[col])
        c = c[c["round"] >= last_round - window]
        return float(c[col].mean()) if len(c) else np.nan

    panels = (["fro"] if have_fro else []) + (["erank"] if erank_cols else [])
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.4), layout="constrained", squeeze=False)
        width = 0.8 / len(zs)
        x = np.arange(len(aggs))
        for ax, panel in zip(axes[0], panels, strict=True):
            for i, z in enumerate(zs):
                pos = x + (i - (len(zs) - 1) / 2) * width
                if panel == "fro":
                    vals = [late(a, z, "agg_fro") / late(a, 0.0, "agg_fro") for a in aggs]
                else:
                    vals = [late(a, z, "erank") for a in aggs]
                ax.bar(pos, vals, width * 0.92, color=colors[z], label=_z_label(z, zf), linewidth=0)
                for p, v in zip(pos, vals, strict=True):
                    if not np.isnan(v):
                        txt = f"{v:.1f}" if panel == "erank" else f"{v:.1f}x"
                        rot = 90 if len(zs) > 4 else 0  # many bars per group: upright labels would collide
                        ax.text(
                            p, v, txt, ha="center", va="bottom", fontsize=8 if rot else 8.5, rotation=rot, color=INK2
                        )
            ax.set_xticks(x, [_agg_name(a) for a in aggs])
            if panel == "fro":
                ax.axhline(1, color=AXIS, linewidth=0.9)
                ax.set_ylabel("Frobenius norm of the aggregate, relative to no attack")
                ax.set_title("Norm of the aggregate", color=INK)
            else:
                ax.set_ylabel("effective rank of the aggregate (mean over layers)")
                ax.set_title("Effective rank of the aggregate", color=INK)
        axes[0][0].legend(frameon=False, loc="upper left", fontsize=9)
        fig.suptitle(
            f"Aggregate received by the server under LIE, average of the last {window} rounds  ({_setup(df)})",
            color=INK,
            x=0.01,
            ha="left",
        )
    return _save(fig, out, "lie_mechanism")


def fig_lie_geometry(df: pd.DataFrame, out: Path, seed: int = 3) -> list[Path]:
    """One coordinate of the update: n−f honest values drawn from N(0,1), f attackers at mean − z·std."""
    n, f = _nf(df)
    zs = _attack_zs(df)
    colors = _z_colors(zs)
    zf = lie_formula_z(n, f)
    rng = np.random.default_rng(seed)
    honest = np.sort(rng.standard_normal(n - f))
    mu, sd = honest.mean(), honest.std()
    rows = [0.0] + zs
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10.5, 2.6 + 0.75 * len(rows)), layout="constrained")
        xs = np.linspace(-3.6, 3.6, 300)
        pdf = np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi) * 1.5
        for i, z in enumerate(rows):
            y = len(rows) - 1 - i
            ax.fill_between(xs, y, y + pdf, color=GRID, lw=0)
            ax.scatter(honest, np.full_like(honest, y + 0.07), s=26, color=INK2, zorder=3)
            if z:
                bad = np.full(f, mu - z * sd)
                ax.scatter(
                    bad + np.linspace(-0.05, 0.05, f),
                    np.full(f, y + 0.07),
                    s=52,
                    color=colors[z],
                    marker="v",
                    zorder=4,
                    edgecolor="white",
                    linewidth=0.5,
                )
                vals = np.concatenate([honest, bad])
            else:
                vals = honest
            stats = {"mean": vals.mean(), "median": float(np.median(vals)), "trimmed_mean": _trimmed(vals, f)}
            for agg, v in stats.items():
                ax.plot([v, v], [y, y + 0.46], color=_agg_color(agg, THREE), linewidth=2.4, zorder=5)
            ax.text(-3.55, y + 0.52, _z_label(z, zf), fontsize=10, color=INK, va="center")
        for agg, v in {"mean": mu, "median": float(np.median(honest)), "trimmed_mean": _trimmed(honest, f)}.items():
            ax.axvline(v, color=_agg_color(agg, THREE), linewidth=0.8, linestyle=":", zorder=1)
        ax.set_ylim(-0.1, len(rows) + 0.25)
        ax.set_yticks([])
        ax.set_xlim(-3.7, 3.7)
        ax.set_xlabel("value of one update coordinate, in honest standard deviations (0 = honest mean)")
        ax.grid(False)
        handles = [
            plt.Line2D([], [], marker="o", color=INK2, linestyle="", markersize=6, label=f"{n - f} honest clients"),
            plt.Line2D([], [], marker="v", color=Z_RAMP[-1], linestyle="", markersize=7, label=f"{f} LIE attackers"),
        ] + [
            plt.Line2D([], [], color=_agg_color(a, THREE), linewidth=2.4, label=f"{_agg_name(a)} of all {n}")
            for a in THREE
        ]
        ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=9, ncol=5, bbox_to_anchor=(1.0, 1.03))
        ax.set_title(
            f"One update coordinate: {n - f} honest values and {f} LIE attackers at (honest mean − z · honest std). "
            "Dotted: aggregates of the honest values alone.",
            color=INK,
            loc="left",
            fontsize=10.5,
        )
    return _save(fig, out, "lie_geometry")


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
    all_aggs = _aggs(df)
    aggs = [a for a in all_aggs if a in THREE]
    attack_zs = _attack_zs(df)
    colors = _z_colors(attack_zs)
    n, f = _nf(df)
    zf = lie_formula_z(n, f)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
        zgrid = np.linspace(0, max(2.5, max(attack_zs) * 1.1), 80)
        sim = simulate_shift(n, f, zgrid, aggs)
        for agg in aggs:
            ax.plot(zgrid, sim[agg], color=_agg_color(agg, all_aggs), linewidth=2.2)
            ax.text(
                zgrid[-1] + 0.04,
                sim[agg][-1],
                _agg_name(agg),
                color=_agg_color(agg, all_aggs),
                va="center",
                fontsize=10,
            )
        top = float(np.nanmax([np.nanmax(v) for v in sim.values()]))
        top = top if np.isfinite(top) and top > 0 else 1.0
        for z in attack_zs:
            ax.axvline(z, color=colors[z], linewidth=1, linestyle=":")
            tag = f"z={z:g}" + (" (paper's formula)" if abs(z - zf) < 1e-3 else "")
            ax.text(z + 0.02, top * 1.02, tag, color=colors[z], fontsize=9, va="top", rotation=90)
        ax.set_xlim(0, zgrid[-1] + 0.6)
        ax.set_ylim(0, top * 1.05)
        ax.set_xlabel("LIE strength z  (attackers send: honest mean − z · honest std)")
        ax.set_ylabel("shift of the aggregate per coordinate\n(in honest standard deviations)")
        ax.set_title(
            f"Shift of the aggregate per coordinate vs. LIE strength  ({f} of {n} clients Byzantine, simulated)",
            color=INK,
            loc="left",
        )
    return _save(fig, out, "lie_shift")


def fig_lie_curves(df: pd.DataFrame, curves: pd.DataFrame, out: Path) -> list[Path]:
    aggs, zs = _aggs(df), _all_zs(df)
    colors = _z_colors(_attack_zs(df))
    zf = lie_formula_z(*_nf(df))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(aggs), figsize=(4.2 * len(aggs), 4.2), sharey=True, squeeze=False)
        for ax, agg in zip(axes[0], aggs, strict=True):
            ends = []
            for z in zs:
                m = _mean_curve(_sel(curves, agg, z), "test_acc")
                if m.empty:
                    continue
                ax.plot(
                    m.index,
                    m.values,
                    color=colors[z],
                    linewidth=2.2 if z else 1.8,
                    linestyle="-" if z else "--",
                    label=_z_label(z, zf),
                )
                ends.append((m.values[-1], m.index[-1], colors[z]))
            ends.sort()
            last_y = -1.0
            for v, r, col in ends:  # direct labels, nudged apart so they never overlap
                y = max(v, last_y + 0.035)
                ax.text(r + 2, y, f"{100 * v:.1f}", color=col, fontsize=9, va="center")
                last_y = y
            ax.set_title(f"aggregator: {_agg_name(agg)}", color=INK)
            ax.set_xlabel("round")
            ax.set_ylim(0, 0.95)
            ax.set_xlim(0, curves["round"].max() * 1.14)
        axes[0][0].set_ylabel("test accuracy")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.02), frameon=False)
        fig.subplots_adjust(bottom=0.22, wspace=0.08, top=0.84, left=0.07, right=0.98)
        fig.suptitle(f"Test accuracy under LIE  ({_setup(df)})", color=INK, x=0.01, ha="left")
    return _save(fig, out, "lie_curves")


def fig_lie_compare(df: pd.DataFrame, out: Path) -> list[Path]:
    """Final accuracy vs z, one panel per aggregator, one line per optimizer: does orthogonalization change the
    dose-response to LIE?"""
    aggs, zs = _aggs(df), _all_zs(df)
    opts = [o for o in OPT_ORDER if o in set(df["opt"])] + sorted(set(df["opt"]) - set(OPT_ORDER))
    g = df.groupby(["opt", "agg", "z"])["final_test_acc"].agg(["mean", "std"])
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(aggs), figsize=(3.9 * len(aggs), 3.9), sharey=True, squeeze=False)
        for ax, agg in zip(axes[0], aggs, strict=True):
            pos = list(range(len(zs)))
            for i, opt in enumerate(opts):
                means = [g["mean"].get((opt, agg, z), np.nan) for z in zs]
                stds = [g["std"].get((opt, agg, z), np.nan) for z in zs]
                ax.errorbar(
                    pos,
                    means,
                    yerr=stds if any(not np.isnan(e) and e > 0 for e in stds) else None,
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    color=CAT[i % len(CAT)],
                    label=OPT_LABEL.get(opt, opt),
                    capsize=2,
                )
            ax.set_xticks(pos, ["none" if z == 0 else f"{z:g}" for z in zs])
            ax.set_xlabel("LIE strength z")
            ax.set_title(f"aggregator: {_agg_name(agg)}", color=INK)
        axes[0][0].set_ylabel(f"test accuracy after {_rounds(df)} rounds")
        axes[0][0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.02), frameon=False)
        fig.subplots_adjust(bottom=0.22, wspace=0.08, top=0.84, left=0.07, right=0.98)
        fig.suptitle(f"Final accuracy vs. LIE strength  ({_setup(df)})", color=INK, x=0.01, ha="left")
    return _save(fig, out, "lie_compare")


def make_lie_figures(results_dir: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """One set of LIE figures per server optimizer (in a sub-folder when there are several) plus, with several
    optimizers, a comparison figure of final accuracy vs z."""
    results_dir = Path(results_dir)
    out = Path(out_dir) if out_dir else results_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df_all = _lie_runs(load_results(results_dir))
    if not df_all["attacked"].any():
        raise ValueError(f"no LIE runs in {results_dir}")
    curves_all = load_curves(results_dir, df_all).merge(df_all[["hash", "z", "attacked"]], on="hash")
    opts = [o for o in OPT_ORDER if o in set(df_all["opt"])] + sorted(set(df_all["opt"]) - set(OPT_ORDER))
    paths: list[Path] = []
    for opt in opts:
        df, curves = df_all[df_all["opt"] == opt], curves_all[curves_all["opt"] == opt]
        if not df["attacked"].any():
            continue
        sub = out if len(opts) == 1 else out / opt
        sub.mkdir(parents=True, exist_ok=True)
        paths += (
            fig_lie_final(df, sub)
            + fig_lie_timeline(df, curves, sub)
            + fig_lie_mechanism(df, curves, sub)
            + fig_lie_geometry(df, sub)
            + fig_lie_shift(df, sub)
            + fig_lie_curves(df, curves, sub)
        )
    if len(opts) > 1:
        paths += fig_lie_compare(df_all, out)
    return paths
