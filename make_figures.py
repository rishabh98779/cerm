"""Generate analytical figures for the revised paper from the simulated dataset."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
GREEN, BLUE, GREY, ORANGE, RED = "#2e7d32", "#1565c0", "#9e9e9e", "#ef6c00", "#c62828"

data = pd.read_csv("simulated_dataset.csv")
res = json.load(open("analysis_output.json"))


def fig_profile():
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    cm = res["profile"]["current_mode"]
    ax[0].bar([c["category"] for c in cm], [c["n"] for c in cm], color=BLUE)
    ax[0].set_title("(a) Current commuting mode (simulated)")
    ax[0].tick_params(axis="x", rotation=30)
    dc = res["profile"]["commute_distance_cat"]
    order = ["<2 km", "2-5 km", "5-10 km", ">10 km"]
    dc = sorted(dc, key=lambda r: order.index(r["category"]))
    ax[1].bar([c["category"] for c in dc], [c["n"] for c in dc], color=GREEN)
    ax[1].set_title("(b) One-way commute distance (simulated)")
    fig.savefig("fig_profile.png")
    plt.close(fig)


def fig_awareness():
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(data.awareness_score, bins=12, color=GREEN, edgecolor="white")
    ax[0].axvline(3, ls="--", color=GREY)
    ax[0].set_title("(a) Awareness score distribution")
    ax[0].set_xlabel("Mean item score (1-5)")
    ax[0].scatter(data.awareness_score, data.intention_score, s=18, color=BLUE, alpha=.7)
    z = np.polyfit(data.awareness_score, data.intention_score, 1)
    xs = np.linspace(data.awareness_score.min(), data.awareness_score.max(), 50)
    ax[1].plot(xs, np.polyval(z, xs), color=RED)
    ax[1].set_title("(b) Awareness vs sustainable intention")
    ax[1].set_xlabel("Awareness"); ax[1].set_ylabel("Intention")
    fig.savefig("fig_awareness.png")
    plt.close(fig)


def fig_mode_share():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels, a, c = [], [], []
    for k, v in res["mode_choice"].items():
        if "code" not in v:
            continue
        labels.append(k); a.append(v["A_pct"]); c.append(v["C_pct"])
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, a, w, label="Stage A (baseline)", color=GREY)
    ax.bar(x + w / 2, c, w, label="Stage C (post-recommendation)", color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("% of simulated decisions"); ax.legend()
    ax.set_title("Baseline vs post-recommendation mode share (simulated)")
    fig.savefig("fig_mode_share.png")
    plt.close(fig)


def fig_co2():
    fig, ax = plt.subplots(figsize=(7, 3.2))
    scen = res["co2"]["by_scenario"]
    keys = list(scen.keys())
    x = np.arange(len(keys)); w = 0.27
    ax.bar(x - w, [scen[k]["all_car"] for k in keys], w, label="All-car reference", color=GREY)
    ax.bar(x, [scen[k]["A"] for k in keys], w, label="Stage A", color=ORANGE)
    ax.bar(x + w, [scen[k]["C"] for k in keys], w, label="Stage C", color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylabel("g CO2 per trip"); ax.legend()
    ax.set_title("Modelled CO2 per trip by scenario (simulated, model-based)")
    fig.savefig("fig_co2.png")
    plt.close(fig)


def fig_projection():
    fig, ax = plt.subplots(figsize=(6, 3.2))
    p = res["projection"]
    labels = ["100%", "50%", "25%"]
    vals = [p["kg_year"], p["half_kg_year"], p["quarter_kg_year"]]
    ax.bar(labels, vals, color=[GREEN, BLUE, ORANGE])
    ax.set_ylabel("kg CO2 saved per commuter / academic year")
    ax.set_xlabel("Assumed share of simulated effect realised")
    ax.set_title("Illustrative CO2 projection (model-based, not observed)")
    fig.savefig("fig_projection.png")
    plt.close(fig)


def fig_method_flow():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axis("off")
    steps = ["Literature review\n(24-source base)", "Research gap", "Research questions\n& hypotheses",
             "Instrument & scenario\ntasks (proposed)", "Simulated dataset\n(N = 120, labelled)",
             "Statistical analysis\n(descriptives, reliability,\ncorrelation, regression)",
             "Baseline vs recommendation\nmode shift", "CO2 model &\nillustrative projection",
             "Five-layer AI framework\n(conceptual)"]
    y = np.linspace(0.95, 0.05, len(steps))
    for i, (s, yy) in enumerate(zip(steps, y)):
        color = "#e8f5e9" if i < 8 else "#e3f2fd"
        ax.add_patch(plt.Rectangle((0.2, yy - 0.04), 0.6, 0.075, facecolor=color, edgecolor=GREEN))
        ax.text(0.5, yy, s, ha="center", va="center", fontsize=8)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y[i + 1] + 0.04), xytext=(0.5, yy - 0.04),
                        arrowprops=dict(arrowstyle="->", color=GREY))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Research process (proposed design with simulated demonstration)")
    fig.savefig("fig_method_flow.png")
    plt.close(fig)


def fig_five_layer():
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axis("off")
    layers = [
        ("Layer 5 - Feedback & Evaluation\n(selected mode, acceptance, CO2, engagement)", "#e3f2fd"),
        ("Layer 4 - Decision & Behavioural Nudging\n(ranking, carbon feedback, gamification, social comparison)", "#e8f5e9"),
        ("Layer 3 - AI Processing\n(route generation, emission estimation, multimodal optimisation)", "#fff3e0"),
        ("Layer 2 - Data Acquisition\n(traffic, transit, weather, air quality, emission factors)", "#f3e5f5"),
        ("Layer 1 - User Input\n(origin, destination, time, preferences, constraints)", "#e0f2f1"),
    ]
    for i, (txt, col) in enumerate(layers):
        ax.add_patch(plt.Rectangle((0.05, 0.08 + i * 0.175), 0.9, 0.15, facecolor=col, edgecolor=GREY))
        ax.text(0.5, 0.155 + i * 0.175, txt, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Five-layer architecture of the proposed commute assistant (conceptual)")
    fig.savefig("fig_five_layer.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_profile(); fig_awareness(); fig_mode_share(); fig_co2()
    fig_projection(); fig_method_flow(); fig_five_layer()
    print("figures written")
