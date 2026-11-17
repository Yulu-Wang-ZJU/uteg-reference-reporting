"""Render Figure 4 from frozen matched operating-point aggregate counts.

The agreement-matched points describe a retrospective comparison, not a
development-selected deployment threshold. See figure4_matched_preview.py for
the calculation and Supplementary Data 4 for aggregate numerators/denominators.
"""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
WS=Path(__file__).resolve().parents[2]
OUT=WS/'outputs/current_ma_20261103/manuscript_sync/deliverables/figures/newB_20261103'
TIMES=[5,10,15]
QUOTAS=[.1,.2,.4]
AGREEMENTS=[.85,.9,.95]
COLORS=[['#396B91','#C1843E','#4B866D'],['#8064A2','#A06072','#60656F']]
MARKERS=[['o','s','^'],['D','v','P']]
def styling():
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .7, "legend.frameon": False})


def plot(points):
    styling()
    lookup = {(r["condition"], r["level"], r["minute"]): r for r in points}
    fig, axes = plt.subplots(2, 2, figsize=(183/25.4, 145/25.4))
    panels = [("a", "coverage", "agreement", "Agreement at matched coverage", "Agreement (%)"),
              ("b", "agreement", "coverage", "Coverage at matched agreement", "Coverage (%)"),
              ("c", "coverage", "fnr", "FNR at matched coverage", "FNR (%)"),
              ("d", "agreement", "fnr", "FNR at matched agreement", "FNR (%)")]
    for ax, (letter, condition, metric, title, ylabel) in zip(axes.flat, panels):
        col = 0 if condition == "coverage" else 1
        levels = QUOTAS if col == 0 else AGREEMENTS
        for i, level in enumerate(levels):
            values = [100*lookup[condition, level, t][metric] if lookup[condition, level, t][metric] is not None else np.nan for t in TIMES]
            ax.plot(TIMES, values, color=COLORS[col][i], marker=MARKERS[col][i],
                markersize=5.6-i*.6, markerfacecolor="none", markeredgewidth=1., lw=1.2)
        ax.set_title(title, loc="left", pad=8)
        ax.text(-.15, 1.07, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.set(xlim=(4.3, 15.7), xticks=TIMES, xlabel="Observation time (min)", ylabel=ylabel)
        if metric == "agreement":
            ax.set(ylim=(80, 102), yticks=[80, 85, 90, 95, 100])
        else:
            ax.set(ylim=(-3, 105), yticks=[0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", color="#E3E7EA", lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.6)
    for col, (levels, label) in enumerate([(QUOTAS, "Fixed coverage (a, c)"), (AGREEMENTS, "Matched agreement (b, d)")]):
        handles = [Line2D([0], [0], color=COLORS[col][i], marker=MARKERS[col][i],
            markerfacecolor="none", markersize=4, lw=1.2, label=f"{level*100:g}%") for i, level in enumerate(levels)]
        fig.legend(handles=handles, title=label, loc="upper center", bbox_to_anchor=(.292+.5*col, .963),
                   ncol=3, fontsize=7, title_fontsize=7.5, columnspacing=1.2, handlelength=1.4)
    fig.subplots_adjust(left=.1, right=.98, top=.81, bottom=.10, hspace=.48, wspace=.3)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"Figure4_main_pair.{ext}", dpi=300, facecolor="white")
    fig.savefig(OUT / 'Figure4_main_pair.tiff', dpi=600, facecolor='white', pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('figure4_matched_aggregate.json')
    points=json.loads(source.read_text(encoding='utf8'))
    assert len(points)==18
    plot(points)
    with (OUT/'Figure4_source_data.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,[k for k in points[0] if k!='interpolation'],extrasaction='ignore');w.writeheader();w.writerows(points)
    qa=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),points=18,paired_a_c_and_b_d=True,matched_observed_agreement=True,retrospective_count_interpolation=True,all_15min_fnr_computed_zero=all(r['fn']==0 and r['true_abnormal']>0 for r in points if r['minute']==15),width_mm=183,height_mm=145)
    (OUT/'Figure4_QA.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf8')
    print('Figure 4 rendered from accepted aggregate counts.')
def run():main()
if __name__=='__main__':main()
