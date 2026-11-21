"""Phase 6, step 5 — static figures (SVG) for the three novelty results.

Palette: validated categorical slots 1-4 (#2a78d6 blue, #eb6834 orange, #1baf7a aqua, #eda100 yellow) on
surface #fcfcfb; aqua/yellow are below 3:1 contrast, so every series also has its own marker and a legend.
Every figure has a CSV table behind it in this folder.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from roadmap.config import PHASE_DIRS  # noqa: E402

DOC = PHASE_DIRS[6]
FIG = DOC / "figures"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 11,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "legend.frameon": False,
    "svg.fonttype": "none",
})


def fig_crude_vs_adjusted():
    d = pd.read_csv(DOC / "DECONFOUNDED_ECG_EFFECTS.csv")
    d = d[d["family"] == "point"]
    style = {"MI-attributable": (BLUE, "o"), "partly confounded": (ORANGE, "s"),
             "explained by confounding": (AQUA, "^"), "revealed by adjustment": (YELLOW, "D"),
             "no association": (GRAY, "x")}
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    lim = 0.52
    ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls="--", zorder=1)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    for cls, (c, m) in style.items():
        s = d[d["class"] == cls]
        ax.scatter(s["crude_sd"], s["att_sd"], s=34 if m != "x" else 30, color=c, marker=m,
                   edgecolors=SURFACE if m != "x" else None, linewidths=1.2 if m != "x" else 1.4,
                   label=f"{cls} ({len(s)})", zorder=3)
    offsets = {"w_ST_Elev_I": (6, -8), "w_T_Amp_aVR": (6, -3), "w_T_Amp_V5": (8, 6)}
    for o, off in offsets.items():
        r = d[d["outcome"] == o.replace("w_", "")].iloc[0]
        ax.annotate(r["outcome"].replace("_Elev", "").replace("_Amp", ""), (r["crude_sd"], r["att_sd"]),
                    xytext=off, textcoords="offset points", fontsize=8, color=INK2)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="Crude difference, MI vs controls (control SD)",
           ylabel="Adjusted effect of MI, ATT (control SD)")
    ax.set_title("ST/T signs survive adjustment; R/Q/S do not")
    ax.text(lim * 0.97, lim * 0.9, "adjusted = crude", ha="right", color=INK2, fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_crude_vs_adjusted.svg")


def fig_st_t_forest():
    d = pd.read_csv(DOC / "DECONFOUNDED_ECG_EFFECTS.csv")
    d = d[(d["family"] == "point") & d["group"].isin(["ST", "T"])].copy()
    d = d.sort_values(["group", "att_sd"]).reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.4, 7.2))
    ax.axvline(0, color=INK2, lw=0.8)
    ax.hlines(y, d["att_sd_ci_low"], d["att_sd_ci_high"], color=BLUE, lw=2, zorder=2)
    ax.scatter(d["att_sd"], y, color=BLUE, s=36, marker="o", edgecolors=SURFACE, linewidths=1.2, zorder=3,
               label="All after-onset MI (primary)")
    ax.scatter(d["nstemi_att_sd"], y + 0.22, color=AQUA, s=30, marker="^", edgecolors=SURFACE, linewidths=1,
               zorder=3, label="NSTEMI only")
    ax.scatter(d["negative_control_att_sd"], y - 0.22, color=ORANGE, s=30, marker="s", edgecolors=SURFACE,
               linewidths=1, zorder=3, label="Negative control: earlier ECG, later MI")
    ax.set_yticks(y, [o.replace("_Elev_", " ").replace("_Amp_", " ") for o in d["outcome"]], fontsize=8)
    split = (d["group"] == "ST").sum() - 0.5
    ax.axhline(split, color=GRID, lw=1.2)
    ax.set_xlabel("Effect of MI on the feature (control SD; bars = 95% CI)")
    ax.set_title("ST and T effects: robust in NSTEMI, far above the negative control")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.07), ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_st_t_effects.svg")


def fig_latent_audit():
    a = pd.read_csv(DOC / "LATENT_AUDIT.csv")
    n_all = len(a)
    a = a[(a["delta_r2_mi_adjusted"] > 0) & (a["r2_baseline"] > 0)]
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    lo, hi = 1e-6, 0.2
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    ax.scatter(a["r2_baseline"], a["delta_r2_mi_adjusted"], color=BLUE, s=30, edgecolors=SURFACE,
               linewidths=1.2, zorder=3)
    for (_, r), off in zip(a.nlargest(2, "delta_r2_mi_adjusted").iterrows(), [(7, 2), (-7, 6)]):
        ax.annotate(f"{r['latent'].replace('z_ecg_', 'z')} (~{r['top_handcrafted_feature'].replace('_Amp_', ' ')})",
                    (r["r2_baseline"], r["delta_r2_mi_adjusted"]), xytext=off, ha="left" if off[0] > 0 else "right",
                    textcoords="offset points", fontsize=8, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
    ax.set(xscale="log", yscale="log", xlim=(3e-4, hi), ylim=(lo, hi),
           xlabel="Variance explained by age, sex and comorbidities (R², cross-fitted)",
           ylabel="Extra variance explained by MI (ΔR², cross-fitted)")
    ax.text(0.12, 0.03, "MI = baseline", ha="right", color=INK2, fontsize=8)
    ax.text(3.5e-4, 1.5e-6, f"{len(a)} of {n_all} latents shown (others have ΔR² or R² ≤ 0)", color=INK2, fontsize=7.5)
    ax.set_title("Latents: MI explains far less than baseline factors")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_latent_audit.svg")


def fig_profiles():
    p = pd.read_csv(DOC / "COUNTERFACTUAL_PROFILES.csv")
    p = p[p["outcome"] == "ST_lateral_mean"].reset_index(drop=True)
    y = np.arange(len(p))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.hlines(y, p["expected_without_mi"], p["expected_with_mi"], color=GRID, lw=2.5, zorder=1)
    ax.scatter(p["expected_without_mi"], y, color=GRAY, s=44, marker="o", edgecolors=SURFACE, linewidths=1.2,
               zorder=3, label="Expected without MI")
    ax.scatter(p["expected_with_mi"], y, color=BLUE, s=44, marker="D", edgecolors=SURFACE, linewidths=1.2,
               zorder=3, label="Expected with MI")
    for yi, (_, r) in zip(y, p.iterrows()):
        ax.annotate(f"{r['effect_sd']:+.2f} SD", (max(r["expected_with_mi"], r["expected_without_mi"]), yi),
                    xytext=(8, -3), textcoords="offset points", ha="left", fontsize=8, color=INK2)
    ax.set_yticks(y, p["profile"], fontsize=8)
    ax.set_xlim(-0.055, 0.03)
    ax.set_ylim(-0.6, len(p) - 0.4)
    ax.set_xlabel("Mean lateral ST level, leads I, aVL, V5, V6 (mV)")
    ax.set_title("Lateral ST shift by profile (model-implied)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_counterfactual_profiles.svg")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_crude_vs_adjusted()
    fig_st_t_forest()
    fig_latent_audit()
    fig_profiles()
    print(sorted(p.name for p in FIG.iterdir()))
