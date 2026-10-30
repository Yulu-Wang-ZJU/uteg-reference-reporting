from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "source_data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
W = 183 * MM
CENTERS = ["Huzhou", "SAHZU", "Hubei"]
SHORT = {"Huzhou": "HZ", "SAHZU": "SA", "Hubei": "HB"}
COLORS = {
    "ink": "#1C2733", "muted": "#5E6B76", "grid": "#E4E9ED",
    "mapped": "#A7B4BE", "adapted": "#1C8A80", "blue": "#2E6FA3",
    "teal": "#1C8A80", "orange": "#D9782B", "red": "#B84F55",
    "plum": "#7D5B8B", "gray": "#BFC8CF", "darkgray": "#6A7780",
}
rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 7.0, "axes.titlesize": 8.0, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
    "axes.linewidth": 0.55, "axes.edgecolor": COLORS["ink"],
    "axes.labelcolor": COLORS["ink"], "xtick.color": COLORS["ink"],
    "ytick.color": COLORS["ink"], "text.color": COLORS["ink"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "pdf.use14corefonts": False,
    "savefig.facecolor": "white",
})


def read_csv(name):
    with (SD / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(v):
    return float(v)


def panel(ax, label, x=-0.13, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def clean_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.45, pad=2)
    if grid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.svg", format="svg", bbox_inches=None, pad_inches=0)
    fig.savefig(OUT / f"{stem}.pdf", format="pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(OUT / f"{stem}.png", format="png", dpi=450, bbox_inches=None, pad_inches=0)
    plt.close(fig)


translation_rows = read_csv("translation_errors.csv")
translation = {c: {} for c in CENTERS}
for row in translation_rows:
    translation[row["center"]].setdefault(row["readout"], {})[row["stage"]] = {
        "value": f(row["MAE"]), "ci95": json.loads(row["ci95"]), "n": int(row["n"])
    }

nmae = {
    "Huzhou": {"value": 0.0915, "lo": 0.0775, "hi": 0.1065, "n": 55, "finite_pairs": 1100},
    "SAHZU": {"value": 0.1077, "lo": 0.0990, "hi": 0.1171, "n": 201, "finite_pairs": 4020},
    "Hubei": {"value": 0.1535, "lo": 0.1194, "hi": 0.1934, "n": 41, "finite_pairs": 820},
}
ma_class = {
    "Huzhou": {"n": 55, "agreement": 71.18, "low_recall": 26.18, "high_recall": None, "support": "17/38/0"},
    "SAHZU": {"n": 201, "agreement": 84.38, "low_recall": 0.00, "high_recall": 0.42, "support": "7/170/24"},
    "Hubei": {"n": 41, "agreement": 38.54, "low_recall": 0.71, "high_recall": 1.94, "support": "7/16/18"},
}

mapping_rows = read_csv("mapping_source_oof.csv")
mapping_order = ["Ridge", "PLS", "KRR", "SVR", "MLP", "TT"]
mapping_counts = {c: {name: 0 for name in mapping_order} for c in CENTERS}
for row in mapping_rows:
    if row["selected"].lower() == "true":
        mapping_counts[row["external_target"]][row["family"]] += 1

ma_gates = {
    "Huzhou": {"MA-ERROR-LR": (75.00, 1.09), "MA-MARGIN": (79.82, 20.73)},
    "SAHZU": {"MA-ERROR-LR": (90.35, 19.08), "MA-MARGIN": (87.46, 37.69)},
    "Hubei": {"MA-ERROR-LR": (math.nan, 0.00), "MA-MARGIN": (40.00, 1.22)},
}
operating = {
    "Huzhou": {"fixed cut": (82.69, 9.45), "balanced": (82.24, 9.73), "Q25": (math.nan, 0.00)},
    "SAHZU": {"fixed cut": (88.12, 62.21), "balanced": (88.29, 58.61), "Q25": (79.14, 20.75)},
    "Hubei": {"fixed cut": (math.nan, 0.00), "balanced": (math.nan, 0.00), "Q25": (71.43, 0.85)},
}
flow = {
    ("Huzhou", "low"): {"n": 17, "correct": 0.00, "false_normal": 0.90, "withheld": 16.10},
    ("SAHZU", "low"): {"n": 7, "correct": 0.00, "false_normal": 4.25, "withheld": 2.75},
    ("SAHZU", "high"): {"n": 24, "correct": 1.00, "false_normal": 9.40, "withheld": 13.60},
    ("Hubei", "low"): {"n": 7, "correct": 0.00, "false_normal": 0.00, "withheld": 7.00},
    ("Hubei", "high"): {"n": 18, "correct": 0.00, "false_normal": 0.00, "withheld": 18.00},
}
matched_hubei = {
    "correct abnormal": {"fixed cut": {"low": 6.67, "high": 5.69}, "Q25": {"low": 62.08, "high": 15.14}},
    "abnormal-to-within": {"fixed cut": {"low": 43.75, "high": 42.22}, "Q25": {"low": 10.00, "high": 23.89}},
}
sequence = json.loads((SD / "SEQUENTIAL.json").read_text(encoding="utf-8"))
sequence = {(r["center"], int(r["seconds"])): r for r in sequence}
unreleased_rows = read_csv("same_5min_unreleased.csv")
unreleased = {(r["center"], int(r["seconds"])): {k: f(v) for k, v in r.items() if k not in {"center", "seconds"}} for r in unreleased_rows}
paired_all = json.loads((SD / "PAIRED_EFFECTS.json").read_text(encoding="utf-8"))
paired = [r for r in paired_all["rows"] if r["family"] == "SEQUENTIAL900_minus_FIXED5"]

used = {
    "nmae": nmae, "translation": translation, "ma_class": ma_class,
    "mapping_counts": mapping_counts, "ma_gates": ma_gates,
    "operating": operating, "flow": {f"{c}_{k}": v for (c, k), v in flow.items()},
    "matched_hubei": matched_hubei, "sequence": list(sequence.values()),
    "same_5min_unreleased": list(unreleased.values()), "paired_sequential": paired,
    "source_files": {"translation": "translation_errors.csv", "clinical": "clinical_translation_metrics.csv", "mapping": "mapping_source_oof.csv", "sequence": "SEQUENTIAL.json", "unreleased": "same_5min_unreleased.csv", "paired": "PAIRED_EFFECTS.json", "fig1": "fig1_corrected_source.png"},
}
(SD / "figure_data_used.json").write_text(json.dumps(used, indent=2, ensure_ascii=False), encoding="utf-8")

# Fig. 1: retain supplied corrected source artwork as raster; do not redraw or upsample it.
# PDF/PNG are replaced by the private corrected PowerPoint export.  This run only
# refreshes the companion SVG wrapper after that export, so a later data rerun
# cannot silently replace the editable-source PDF/PNG with a matplotlib redraw.
fig = plt.figure(figsize=(W, 147 * MM), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
img = Image.open(SD / "fig1_corrected_source.png")
ax.imshow(img, interpolation="none")
ax.axis("off")
fig.savefig(OUT / "Figure1_workflow.svg", format="svg", bbox_inches=None, pad_inches=0)
plt.close(fig)

# Fig. 2: translation, readout error and source-side selection.
fig = plt.figure(figsize=(W, 191 * MM), facecolor="white")
gs = fig.add_gridspec(2, 2, left=0.075, right=0.975, bottom=0.105, top=0.93, wspace=0.34, hspace=0.42)

ax = fig.add_subplot(gs[0, 0])
panel(ax, "a")
x = list(range(3))
means = [nmae[c]["value"] for c in CENTERS]
lo = [means[i] - nmae[c]["lo"] for i, c in enumerate(CENTERS)]
hi = [nmae[c]["hi"] - means[i] for i, c in enumerate(CENTERS)]
ax.errorbar(x, means, yerr=[lo, hi], fmt="o", color=COLORS["blue"], ecolor=COLORS["blue"], capsize=2.5, markersize=4, lw=0.8)
ax.set_xticks(x, [SHORT[c] for c in CENTERS])
ax.set_ylim(0, 0.22)
ax.set_ylabel("NMAE (model-constrained curves)")
ax.set_title("Model-constrained curve discrepancy", loc="left", pad=4)
clean_axis(ax)

sub = gs[0, 1].subgridspec(2, 2, wspace=0.38, hspace=0.72)
readouts = [("R_min", "R MAE (min)", 3.0), ("K_min", "K MAE (min)", 1.5), ("MA_mm", "MA MAE (mm)", 15.0), ("angle_deg", "Angle MAE (deg)", 12.0)]
for j, (key, ylabel, ymax) in enumerate(readouts):
    axb = fig.add_subplot(sub[j // 2, j % 2])
    if j == 0:
        panel(axb, "b", x=-0.30, y=1.17)
    xx = list(range(3))
    mapped = [translation[c][key]["Mapped"]["value"] for c in CENTERS]
    adapted = [translation[c][key]["Adapted"]["value"] for c in CENTERS]
    width = 0.32
    axb.bar([v - width / 2 for v in xx], mapped, width, color=COLORS["mapped"], label="mapped" if j == 0 else None, zorder=2)
    axb.bar([v + width / 2 for v in xx], adapted, width, color=COLORS["adapted"], label="adapted" if j == 0 else None, zorder=2)
    axb.set_xticks(xx, [SHORT[c] for c in CENTERS])
    axb.set_ylim(0, ymax)
    axb.set_title(ylabel, fontsize=6.8, loc="left", pad=3)
    axb.tick_params(axis="y", labelsize=5.6)
    clean_axis(axb)
    if j == 0:
        axb.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8)
ax = fig.add_subplot(gs[1, 0])
panel(ax, "c")
agreement = [ma_class[c]["agreement"] for c in CENTERS]
ax.bar(x, agreement, color=COLORS["blue"], alpha=0.85, width=0.48, label="MA agreement", zorder=2)
for i, c in enumerate(CENTERS):
    ax.plot(i - 0.13, ma_class[c]["low_recall"], marker="o", ms=4, color=COLORS["orange"], label="low recall" if i == 0 else None, zorder=4)
    if ma_class[c]["high_recall"] is None:
        ax.plot(i + 0.13, 2.0, marker="_", ms=8, color=COLORS["red"], zorder=4)
        ax.text(i + 0.13, 4.0, "NA", ha="center", fontsize=5.8, color=COLORS["red"])
    else:
        ax.plot(i + 0.13, ma_class[c]["high_recall"], marker="o", ms=4, color=COLORS["red"], label="high recall" if i == 1 else None, zorder=4)
    ax.text(i, 102, f"n={ma_class[c]['n']}", ha="center", va="bottom", fontsize=5.5, color=COLORS["muted"])
ax.set_xticks(x, [SHORT[c] for c in CENTERS])
ax.set_ylim(-2, 116)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel("Percent (%)")
ax.set_title("Frozen 15-min MA state preservation", loc="left", pad=4)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.22), ncol=3, handlelength=1.2, columnspacing=0.8)
clean_axis(ax)

ax = fig.add_subplot(gs[1, 1])
panel(ax, "d")
bottom = [0, 0, 0]
family_colors = ["#C7D0D6", "#B4C3CF", "#9BB2C2", "#6E93A8", COLORS["teal"], COLORS["orange"]]
for fam, color in zip(mapping_order, family_colors):
    vals = [mapping_counts[c][fam] for c in CENTERS]
    ax.bar(x, vals, bottom=bottom, color=color, width=0.58, label=fam, edgecolor="white", linewidth=0.25)
    bottom = [bottom[i] + vals[i] for i in range(3)]
ax.set_xticks(x, [SHORT[c] for c in CENTERS])
ax.set_ylim(0, 20)
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_ylabel("Selected seeds (n=20)")
ax.set_title("Source-side grouped OOF family selection", loc="left", pad=4)
ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.20), handlelength=1.0, columnspacing=0.8)
clean_axis(ax)
save(fig, "Figure2_translation")

# Fig. 3: selective reporting and amplitude-policy trade-offs.
fig = plt.figure(figsize=(W, 190 * MM), facecolor="white")
gs = fig.add_gridspec(2, 2, left=0.075, right=0.975, bottom=0.105, top=0.93, wspace=0.34, hspace=0.45)

sub = gs[0, 0].subgridspec(2, 1, hspace=0.72)
ax_ag = fig.add_subplot(sub[0, 0])
panel(ax_ag, "a")
ax_cov = fig.add_subplot(sub[1, 0], sharex=ax_ag)
for axx, metric, ylabel in [(ax_ag, 0, "Agreement (%)"), (ax_cov, 1, "Coverage (%)")]:
    for gate, color, marker, offset in [("MA-ERROR-LR", COLORS["orange"], "o", -0.12), ("MA-MARGIN", COLORS["blue"], "s", 0.12)]:
        vals = [ma_gates[c][gate][metric] for c in CENTERS]
        xx = [i + offset for i in x]
        finite = [i for i, val in enumerate(vals) if not math.isnan(val)]
        axx.scatter([xx[i] for i in finite], [vals[i] for i in finite], color=color, marker=marker, s=16, label=gate if metric == 0 else None, zorder=3)
    axx.set_ylim(0, 100)
    axx.set_ylabel(ylabel)
    clean_axis(axx)
ax_ag.set_title("Frozen MA gates across centers", loc="left", pad=4)
ax_ag.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.02, 0.02), handlelength=1.2)
ax_cov.set_xticks(x, [SHORT[c] for c in CENTERS])
ax_cov.set_xlabel("Target center")
ax_ag.tick_params(labelbottom=False)
ax_ag.text(2 - 0.12, 92, "NA", fontsize=6, color=COLORS["orange"], ha="center")

ax = fig.add_subplot(gs[0, 1])
panel(ax, "b")
policy_style = {"fixed cut": (COLORS["blue"], "o"), "balanced": (COLORS["teal"], "s"), "Q25": (COLORS["orange"], "^")}
for c in CENTERS:
    for pol, (agree, cov) in operating[c].items():
        color, marker = policy_style[pol]
        if not math.isnan(agree):
            ax.plot(cov, agree, marker=marker, ms=4.5, color=color, markerfacecolor=color)
    # One center tag per target keeps the policy markers readable without stacking duplicate HZ labels.
    ax.text(11.0, 87.0, "HZ", fontsize=5.4, color=COLORS["ink"])
    ax.text(21.5, 83.0, "SA", fontsize=5.4, color=COLORS["ink"])
    ax.text(63.0, 89.5, "SA", fontsize=5.4, color=COLORS["ink"])
    ax.text(2.2, 74.5, "HB", fontsize=5.4, color=COLORS["ink"])
ax.set_xlim(-2, 68)
ax.set_ylim(0, 100)
ax.set_xlabel("Coverage (%) · full test denominator")
ax.set_ylabel("Agreement (%)")
ax.set_title("Frozen amplitude-policy operating points", loc="left", pad=4)
ax.legend(handles=[Line2D([0], [0], marker=m, color=c, linestyle="", label=p, markersize=4.5) for p, (c, m) in policy_style.items()], frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.22), handlelength=1.0)
clean_axis(ax)

ax = fig.add_subplot(gs[1, 0])
panel(ax, "c")
bar_keys = [("Huzhou", "low"), ("SAHZU", "low"), ("SAHZU", "high"), ("Hubei", "low"), ("Hubei", "high")]
x2 = list(range(len(bar_keys)))
bottom = [0] * len(x2)
for name, field, color in [("correct abnormal", "correct", COLORS["blue"]), ("abnormal→within", "false_normal", COLORS["orange"]), ("withheld", "withheld", COLORS["gray"])]:
    heights = [100 * flow[k][field] / flow[k]["n"] for k in bar_keys]
    ax.bar(x2, heights, bottom=bottom, width=0.68, color=color, label=name, edgecolor="white", linewidth=0.3)
    bottom = [bottom[i] + heights[i] for i in range(len(x2))]
for i, key in enumerate(bar_keys):
    ax.text(i, 102, f"{SHORT[key[0]]} {key[1]}\nn={flow[key]['n']}", ha="center", va="bottom", fontsize=5.3)
ax.set_xticks(x2, ["low", "low", "high", "low", "high"])
ax.set_ylim(0, 118)
ax.set_ylabel("Reference-class share (%)")
ax.set_title("Amplitude fixed-cut disposition", loc="left", pad=4)
ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.23), handlelength=1.2, columnspacing=0.7)
clean_axis(ax)

ax = fig.add_subplot(gs[1, 1])
panel(ax, "d")
groups = [("low", "correct abnormal"), ("high", "correct abnormal"), ("low", "abnormal-to-within"), ("high", "abnormal-to-within")]
x3 = list(range(len(groups)))
for offset, pol in [(-0.18, "fixed cut"), (0.18, "Q25")]:
    vals = [matched_hubei[outcome][pol][cls] for cls, outcome in groups]
    ax.bar([v + offset for v in x3], vals, width=0.34, color=COLORS["blue"] if pol == "fixed cut" else COLORS["orange"], label=pol, zorder=2)
    for i, val in enumerate(vals):
        ax.text(i + offset, val + 1.5, f"{val:.2f}", ha="center", fontsize=5.1, color=COLORS["ink"])
ax.set_xticks(x3, ["low\ncorrect", "high\ncorrect", "low\nabnormal→within", "high\nabnormal→within"])
ax.set_ylim(0, 72)
ax.set_ylabel("Share of reference class (%)")
ax.set_title("Hubei matched 50% coverage · QC n=39", loc="left", pad=4)
ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.22), handlelength=1.2)
clean_axis(ax)
save(fig, "Figure3_reporting")

# Fig. 4: secondary CI continuation sequence.
fig = plt.figure(figsize=(W, 190 * MM), facecolor="white")
gs = fig.add_gridspec(2, 2, left=0.075, right=0.975, bottom=0.105, top=0.93, wspace=0.35, hspace=0.47)

ax = fig.add_subplot(gs[0, 0])
panel(ax, "a")
x4 = []
labels = []
for ci, c in enumerate(CENTERS):
    for ti, sec in enumerate([300, 600, 900]):
        x4.append(ci * 4 + ti)
        labels.append(str(sec // 60))
    if ci < 2:
        ax.axvline(ci * 4 + 3.45, color=COLORS["grid"], lw=0.7)
bottom = [0] * len(x4)
for field, color, name in [("correct_yield", COLORS["blue"], "correct report"), ("wrong_yield", COLORS["red"], "incorrect report"), ("no_report", COLORS["gray"], "unreported")]:
    vals = [100 * sequence[(c, sec)][field] for c in CENTERS for sec in [300, 600, 900]]
    ax.bar(x4, vals, bottom=bottom, width=0.72, color=color, label=name, edgecolor="white", linewidth=0.3)
    bottom = [bottom[i] + vals[i] for i in range(len(x4))]
ax.set_xticks(x4, labels)
ax.set_ylim(0, 100)
ax.set_xlabel("Observation time (min)")
ax.set_ylabel("Share of full test cohort (%)")
ax.set_title("Cumulative CI first-report states", loc="left", pad=4)
ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.22), handlelength=1.2, columnspacing=0.7)
clean_axis(ax)
for ci, c in enumerate(CENTERS):
    ax.text(ci * 4 + 1, 104, SHORT[c], ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=COLORS["ink"])
ax.set_ylim(0, 115)

sub = gs[0, 1].subgridspec(2, 1, hspace=0.62)
for idx, (field, title, ylabel) in enumerate([("coverage", "Cumulative coverage", "Coverage (%)"), ("agreement", "Selected agreement among reports", "Agreement (%)")]):
    axb = fig.add_subplot(sub[idx, 0])
    if idx == 0:
        panel(axb, "b")
    for c in CENTERS:
        vals = [100 * sequence[(c, sec)][field] for sec in [300, 600, 900]]
        axb.plot([5, 10, 15], vals, marker="o", ms=3.2, lw=0.9, color={"Huzhou": COLORS["blue"], "SAHZU": COLORS["teal"], "Hubei": COLORS["orange"]}[c], label=SHORT[c])
    axb.set_xticks([5, 10, 15])
    axb.set_ylim(0, 100)
    axb.set_ylabel(ylabel)
    axb.set_title(title, fontsize=6.8, loc="left", pad=2)
    clean_axis(axb)
    if idx == 0:
        axb.legend(frameon=False, ncol=3, loc="upper left", handlelength=1.2)

ax = fig.add_subplot(gs[1, 0])
panel(ax, "c")
x5 = []
labels5 = []
for ci, c in enumerate(CENTERS):
    for ti, sec in enumerate([600, 900]):
        x5.append(ci * 3 + ti)
        labels5.append(str(sec // 60))
    if ci < 2:
        ax.axvline(ci * 3 + 1.45, color=COLORS["grid"], lw=0.7)
bottom = [0] * len(x5)
for key, color, name in [("new_correct_n", COLORS["blue"], "new correct"), ("new_false_normal_n", COLORS["orange"], "abnormal-to-within"), ("new_other_error_n", COLORS["plum"], "other wrong"), ("still_no_report_n", COLORS["gray"], "still withheld")]:
    vals = [unreleased[(c, sec)][key] for c in CENTERS for sec in [600, 900]]
    ax.bar(x5, vals, bottom=bottom, width=0.72, color=color, label=name, edgecolor="white", linewidth=0.3)
    bottom = [bottom[i] + vals[i] for i in range(len(x5))]
ax.set_xticks(x5, labels5)
ax.set_ylabel("Mean records per run")
ax.set_xlabel("Later time point · same records unreported at 5 min")
ax.set_title("Follow-up outcomes among the 5-min unreported cohort", loc="left", pad=4)
ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.22), handlelength=1.1, columnspacing=0.5)
clean_axis(ax)
for ci, c in enumerate(CENTERS):
    ax.text(ci * 3 + 0.5, 106, SHORT[c], ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=COLORS["ink"])
ax.set_ylim(0, 115)

ax = fig.add_subplot(gs[1, 1])
panel(ax, "d")
rows = []
for c in CENTERS:
    for metric in ["correctyield", "wrongyield"]:
        rows.append((c, metric, next(r for r in paired if r["target"] == c and r["metric"] == metric)))
ypos = list(range(len(rows)))[::-1]
for y, (c, metric, row) in zip(ypos, rows):
    point = 100 * row["point"]
    ci = [100 * v for v in row["ci95"]]
    bonf = [100 * v for v in row["ci_bonf6"]]
    color = COLORS["blue"] if metric == "correctyield" else COLORS["red"]
    ax.hlines(y, ci[0], ci[1], color=color, lw=0.9, alpha=0.65)
    ax.hlines(y, bonf[0], bonf[1], color=color, lw=2.5, alpha=0.8)
    ax.plot(point, y, "o", color=color, ms=4, zorder=3)
ax.axvline(0, color=COLORS["ink"], lw=0.55)
ax.set_yticks(ypos, [f"{SHORT[c]} {'correct' if m == 'correctyield' else 'wrong'}" for c, m, _ in rows])
ax.set_xlabel("15 min − 5 min (percentage points)")
ax.set_title("15 min versus stopping at 5 min", loc="left", pad=4)
ax.set_xlim(-2, 40)
clean_axis(ax, grid=False)
ax.legend(handles=[Line2D([0], [0], color=COLORS["blue"], lw=0.9, label="95% interval"), Line2D([0], [0], color=COLORS["blue"], lw=2.5, label="Bonferroni-6 interval")], frameon=False, loc="lower right", handlelength=1.5)
save(fig, "Figure4_observation")

print(json.dumps({"output": str(OUT), "figures": sorted(p.name for p in OUT.iterdir())}, ensure_ascii=False, indent=2))
