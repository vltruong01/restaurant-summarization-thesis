"""Render only the two requested Chapter 4 figures using supplied values."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/chapter4_figures_20261112_v1"
DPI = 600
COLORS = {"U": "#8795A1", "F": "#326B7B", "S": "#B8BEC3"}
SYSTEM_COLORS = ["#326B7B", "#8795A1"]


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3.5)
    ax.grid(axis="y", color="#E3E6E8", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.20)


def labels(ax, bars, values, decimals, offset):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )


def save(fig, filename, expected_size):
    target = OUT / filename
    if target.exists():
        raise FileExistsError(f"Existing figure preserved: {target}")
    fig.savefig(target, dpi=DPI, facecolor="white")
    plt.close(fig)
    with Image.open(target) as png:
        png.verify()
    with Image.open(target) as png:
        assert png.size == expected_size, (png.size, expected_size)
        assert all(abs(dpi - DPI) < 0.1 for dpi in png.info["dpi"])
    print(f"{target.name}: {expected_size[0]} x {expected_size[1]} pixels, {DPI} dpi")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.titleweight": "normal",
            "text.color": "#202020",
            "axes.labelcolor": "#202020",
            "xtick.color": "#202020",
            "ytick.color": "#202020",
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.13, top=0.81)
    fig.suptitle(
        "Figure 4.1. Human Evaluation of Visual Grounding and Contribution\n"
        "on Yelp TEST30",
        x=0.5,
        y=0.94,
        fontsize=12,
        linespacing=1.5,
    )
    values = [4.00, 3.90, 1.23]
    bars = ax.bar(
        ["U", "F", "S"],
        values,
        width=0.56,
        color=[COLORS[x] for x in ["U", "F", "S"]],
        edgecolor="#495057",
        linewidth=0.5,
    )
    ax.set_ylabel("Mean human rating (1–5)")
    ax.set_ylim(0, 5)
    ax.set_yticks(range(6))
    labels(ax, bars, values, decimals=2, offset=0.09)
    style_axes(ax)
    assert ax.get_ylim() == (0, 5)
    save(fig, "Figure_4_1_Visual_Grounding_Yelp_TEST30.png", (5040, 3360))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.22, top=0.86, wspace=0.35)
    names = ["LED F-S1000", "Qwen-Caption"]
    panels = [
        ("(a) Generation time", [97.931, 186.844], "Generation time (seconds)", 225, 5),
        (
            "(b) Peak allocated GPU memory",
            [2.697, 18.536],
            "Peak allocated GPU memory (GiB)",
            22.5,
            0.5,
        ),
    ]
    for ax, (title, values, ylabel, ceiling, offset) in zip(axes, panels):
        bars = ax.bar(
            names,
            values,
            width=0.55,
            color=SYSTEM_COLORS,
            edgecolor="#495057",
            linewidth=0.5,
        )
        ax.set_title(title, pad=17)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ceiling)
        ax.set_yticks([0, 50, 100, 150, 200] if ax is axes[0] else [0, 5, 10, 15, 20])
        labels(ax, bars, values, decimals=3, offset=offset)
        style_axes(ax)
    fig.text(
        0.5,
        0.065,
        "Prepared inputs only; excludes upstream BLIP-2 caption generation and selection.",
        ha="center",
        va="center",
        fontsize=10,
    )
    save(fig, "Figure_4_2_LED_vs_Qwen_Combined.png", (6600, 3000))
    assert len(list(OUT.glob("*.png"))) == 2


if __name__ == "__main__":
    main()
