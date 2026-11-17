import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, GREY, INK, MUTED, GRID = "#2a78d6", "#9a988f", "#2b2b2a", "#6b6a64", "#ecebe7"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": "#c9c8c2", "xtick.color": MUTED, "ytick.color": INK})
cv = json.load(open("cv100.json")); cv["Head ablation + attention"] = [0.531, 0.502, 0.549]
internal = {"Attention to repeated words", "Head-ablation profile", "Syntactic-role probe", "Head ablation + attention"}
order = ["MNLI accuracy", "ID behaviour summary", "Weight geometry (Euclidean)", "Convexity-gap kernel (Juneja)",
         "Weight deltas", "Representations (CKA)", "ID high-overlap items", "All baselines",
         "Syntactic-role probe", "Attention to repeated words", "Head-ablation profile", "Head ablation + attention"][::-1]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), dpi=300, gridspec_kw={"width_ratios": [1.55, 1]})
for i, k in enumerate(order):
    m, lo, hi = cv[k]; c = BLUE if k in internal else GREY
    a.barh(i, m, height=0.62, color=c)
    a.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5)
    a.text(hi + 0.012, i, f"{m:.2f}", va="center", fontsize=6.5, color=INK)
a.set_yticks(range(len(order))); a.set_yticklabels(order, fontsize=6.8)
a.set_xlim(0, 0.75); a.set_xlabel("CV $R^2$ (100 models, 10 x 10-fold)", color=INK)
a.set_title("(a) Each signal alone", fontsize=8, color=INK, loc="left")
a.xaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
for s in ("top", "right"): a.spines[s].set_visible(False)
inc = [("Head ablation + attention\nover baselines", -0.0122, -0.0354, 0.0110, BLUE),
       ("Head ablation + attention\nover baselines + CG", -0.0068, -0.0246, 0.0110, BLUE),
       ("Syntactic probe\nover baselines + CG", -0.0035, -0.0191, 0.0121, BLUE),
       ("Convexity gap (CG)\nover baselines", 0.0143, -0.0586, 0.0873, GREY),
       ("Pure-noise block\n(calibration, mean of 5)", -0.0020, -0.0085, 0.0045, "#c9c8c2")][::-1]
for i, (lab, m, lo, hi, c) in enumerate(inc):
    b.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, ecolor=c, ms=4, elinewidth=1.2, capsize=2)
b.axvline(0, color=INK, lw=0.8); b.axvline(0.10, color=MUTED, lw=0.8, ls="--")
b.text(0.097, 0.15, "pre-registered\nGO threshold", fontsize=6, color=MUTED, ha="right", va="bottom")
b.set_yticks(range(len(inc))); b.set_yticklabels([x[0] for x in inc], fontsize=6.5)
b.set_xlim(-0.08, 0.12); b.set_xlabel("Gain in held-out $R^2$ (95% CI)", color=INK)
b.set_title("(b) Added to the baselines", fontsize=8, color=INK, loc="left")
b.xaxis.grid(True, color=GRID, lw=0.6); b.set_axisbelow(True)
for s in ("top", "right"): b.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("fig_main.pdf"); plt.savefig("fig_main.png")

f = json.load(open("fragility.json"))
fig, ax = plt.subplots(figsize=(3.3, 2.4), dpi=300)
xs = [100 * v for v in f["mnli"]]; ys = [100 * v for v in f["acc_drops"]]
rnd = [n.startswith("rand") for n in f["names"]]
ax.scatter([x for x, r in zip(xs, rnd) if r], [y for y, r in zip(ys, rnd) if r], s=10, color=GREY, label="random 6-head sets (40)", zorder=3)
ax.scatter([x for x, r in zip(xs, rnd) if not r], [y for y, r in zip(ys, rnd) if not r], s=10, color=GREY, marker="^", label="depth-matched sets (20)", zorder=3)
ax.scatter([-0.9], [10.8], s=22, color=BLUE, label="six most predictive heads", zorder=4)
ax.axhline(0, color=INK, lw=0.7); ax.axvline(0, color=INK, lw=0.7)
ax.set_xlabel("Change in MNLI accuracy (points)", color=INK); ax.set_ylabel("Drop in HANS-LO accuracy (points)", color=INK)
ax.set_xlim(-1.5, 1.5); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_ylim(-4.5, 15.5); ax.legend(fontsize=6, frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig("fig_fragility.pdf"); plt.savefig("fig_fragility.png")
