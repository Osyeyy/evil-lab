"""Plots (matplotlib, Agg backend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREY, RED, GREEN = "#2a6fdb", "#e8833a", "#9a9a9a", "#c8423b", "#3c9d5d"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def layer_separability(stats, path):
    L = [s["layer"] for s in stats]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    ax[0].plot(L, [s["heldout_auroc"] for s in stats], color=BLUE, marker="o", ms=3)
    ax[0].set(title="Held-out AUROC of projection", xlabel="layer", ylim=(0.4, 1.02))
    ax[1].plot(L, [s["split_half_cos"] for s in stats], color=BLUE, marker="o", ms=3)
    ax[1].set(title="Split-half cosine of direction", xlabel="layer")
    ax[2].plot(L, [s["rel_diff_norm"] for s in stats], color=BLUE, marker="o", ms=3)
    ax[2].set(title="|mean diff| / mean residual norm", xlabel="layer")
    for a in ax:
        a.grid(alpha=0.3)
    _save(fig, path)


def screen_plot(screen, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    coefs = sorted({r["coef"] for r in screen["rows"]})
    for i, c in enumerate(coefs):
        rows = [r for r in screen["rows"] if r["coef"] == c]
        ls = [r["layer"] for r in rows]
        alpha = 0.5 + 0.5 * (i + 1) / len(coefs)
        ax.plot(ls, [r["delta_pref"] for r in rows], color=BLUE, alpha=alpha, marker="o", ms=3, label=f"persona dir c={c}")
        ax.plot(ls, [r["delta_pref_random"] for r in rows], color=GREY, alpha=alpha, ls="--", label=f"random dir c={c}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(title="Steering effect on logprob preference (dev, no system prompt)", xlabel="layer",
           ylabel="delta mean logprob (trait - anti)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    _save(fig, path)


def sweep_plot(sweep, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    layers = sorted({g["layer"] for g in sweep["grid"]})
    cmap = plt.get_cmap("viridis")
    for i, l in enumerate(layers):
        rows = sorted([g for g in sweep["grid"] if g["layer"] == l], key=lambda g: g["coef"])
        col = cmap(i / max(1, len(layers) - 1))
        ax[0].plot([g["coef"] for g in rows], [g["trait"] for g in rows], marker="o", color=col, label=f"L{l}")
        ax[1].plot([g["coef"] for g in rows], [g["coherence"] for g in rows], marker="o", color=col, label=f"L{l}")
    b = sweep["baseline"]
    ax[0].axhline(b["trait_mean"], color=GREY, ls="--", label="baseline")
    ax[1].axhline(b["coherence_mean"], color=GREY, ls="--")
    ax[0].set(title="Trait score vs steering strength (dev)", xlabel="coef (x mean resid norm)", ylabel="trait 0-100")
    ax[1].set(title="Coherence vs steering strength (dev)", xlabel="coef", ylabel="coherence 0-100")
    if sweep.get("best"):
        ax[0].scatter([sweep["best"]["coef"]], [sweep["best"]["trait"]], s=140, facecolors="none", edgecolors=RED, lw=2)
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    _save(fig, path)


def patching_plot(trace, path):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(trace["layers"], [x["recovery"] for x in trace["suffix"]], marker="o", ms=3, color=BLUE, label="patch user-turn suffix")
    ax.plot(trace["layers"], [x["recovery"] for x in trace["last"]], marker="o", ms=3, color=ORANGE, label="patch last prompt token")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(1, color=GREY, lw=0.6, ls="--")
    ax.set(title="Causal tracing: recovery of trait preference", xlabel="layer patched", ylabel="fraction of gap recovered")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, path)


def conditions_plot(summaries, path):
    names = [s["condition"] for s in summaries]
    tm = [s["trait_mean"] for s in summaries]
    lo = [s["trait_mean"] - s["trait_ci95"][0] for s in summaries]
    hi = [s["trait_ci95"][1] - s["trait_mean"] for s in summaries]
    coh = [s["coherence_mean"] for s in summaries]
    cols = [GREY if n in ("baseline", "sham") or n.startswith("random") else
            (ORANGE if n.startswith("sysprompt") else BLUE) for n in names]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 4.2))
    x = np.arange(len(names))
    ax.bar(x, tm, color=cols, yerr=[lo, hi], capsize=3)
    ax.plot(x, coh, "D", color=GREEN, label="coherence")
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8)
    ax.set(ylabel="score 0-100", title="Held-out test: trait expression by condition (bars, 95% CI) and coherence (diamonds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, path)


def neuron_plot(neurons, path):
    t = np.asarray(neurons["t"], dtype=np.float32)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ax[0].plot(neurons["frac_significant_per_layer"], color=BLUE, marker="o", ms=3)
    ax[0].set(title="Fraction of MLP neurons with |t| > 5", xlabel="layer")
    ax[1].hist(t.flatten(), bins=200, color=BLUE, log=True)
    ax[1].set(title="Welch t (trait vs anti-trait), all neurons", xlabel="t")
    for a in ax:
        a.grid(alpha=0.3)
    _save(fig, path)


def heads_plot(heads, path):
    c = np.asarray(heads["contrib_diff"], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 4))
    m = np.abs(c).max()
    im = ax.imshow(c.T, aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m)
    fig.colorbar(im, ax=ax)
    ax.set(title="Head contribution along persona direction (trait - anti)", xlabel="layer", ylabel="head")
    _save(fig, path)


def compare_plot(rows, path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for r in rows:
        stats = r["stats"]
        n = len(stats)
        ax.plot([s["layer"] / (n - 1) for s in stats], [s["heldout_auroc"] for s in stats], label=r["label"])
        if r.get("best_layer") is not None:
            ax.axvline(r["best_layer"] / (n - 1), ls=":", alpha=0.6)
    ax.set(title="Persona separability across models (relative depth)", xlabel="relative depth", ylabel="held-out AUROC")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    _save(fig, path)
