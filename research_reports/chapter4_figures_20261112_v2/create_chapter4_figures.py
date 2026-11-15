"""Reproduce the two Chapter 4 figures, as PNG (600 DPI) and vector PDF.

Requirements: Python 3.10+, matplotlib, Pillow.
Run: python create_chapter4_figures.py
Files are written beside this script. No thesis or other research file is read
or modified. Numeric constants match the researcher-supplied rounded values.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


OUTPUT = Path(__file__).resolve().parent
DPI = 600
TEAL = "#326B7B"
SLATE = "#8795A1"
LIGHT_SLATE = "#B8BEC3"


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 9.5,
            "axes.titleweight": "normal",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.labelcolor": "#202020",
            "text.color": "#202020",
            "xtick.color": "#202020",
            "ytick.color": "#202020",
            "axes.edgecolor": "#404040",
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            # Embedded TrueType text; bars and axes remain vector paths.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=0.7, length=3.0)
    ax.grid(False)
    ax.margins(x=0.20)


def draw_bars(ax, names, values, colors, decimals, offset):
    bars = ax.bar(names, values, width=0.56, color=colors, edgecolor="none")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )
    style_axes(ax)
    assert [bar.get_height() for bar in bars] == values
    assert not ax.lines, "No error bars or significance markers are requested."


def export(fig, stem):
    """Use identical physical dimensions for the raster and vector versions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            assert bounds.x0 >= canvas.x0 - 1 and bounds.x1 <= canvas.x1 + 1
            assert bounds.y0 >= canvas.y0 - 1 and bounds.y1 <= canvas.y1 + 1
    assert not fig.texts, "No overall title, caption or explanatory footer."
    expected_size = tuple(int(round(value * DPI)) for value in fig.get_size_inches())
    png = OUTPUT / f"{stem}.png"
    pdf = OUTPUT / f"{stem}.pdf"
    fig.savefig(png, dpi=DPI)
    fig.savefig(pdf, format="pdf", metadata={"Creator": "Python / matplotlib"})
    plt.close(fig)
    with Image.open(png) as image:
        assert image.size == expected_size
        assert all(abs(value - DPI) < 0.1 for value in image.info["dpi"])
        image.verify()
    print(f"{png.name}: {expected_size[0]} x {expected_size[1]} pixels, {DPI} DPI")
    print(f"{pdf.name}: vector PDF with embedded TrueType text")


def figure_4_1():
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.14, top=0.97)
    ax.set_ylabel("Mean human rating (1–5)", labelpad=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    draw_bars(
        ax,
        ["U", "F", "S"],
        [4.00, 3.90, 1.23],
        [SLATE, TEAL, LIGHT_SLATE],
        decimals=2,
        offset=0.085,
    )
    assert ax.get_ylim() == (0, 5) and ax.get_title() == ""
    export(fig, "Figure_4_1_Visual_Grounding_Yelp_TEST30")


def figure_4_2():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.5))
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.16, top=0.86, wspace=0.48)
    panels = [
        (
            "(a) Generation time",
            [97.931, 186.844],
            "Generation time (seconds)",
            225,
            [0, 50, 100, 150, 200],
            4.5,
        ),
        (
            "(b) Peak allocated GPU memory",
            [2.697, 18.536],
            "Peak allocated GPU memory (GiB)",
            22.5,
            [0, 5, 10, 15, 20],
            0.45,
        ),
    ]
    for ax, (title, values, ylabel, ymax, ticks, offset) in zip(axes, panels):
        ax.set_title(title, pad=12)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_ylim(0, ymax)
        ax.set_yticks(ticks)
        draw_bars(
            ax,
            ["LED F-S1000", "Qwen-Caption"],
            values,
            [TEAL, SLATE],
            decimals=3,
            offset=offset,
        )
        assert ax.get_ylim()[0] == 0
    export(fig, "Figure_4_2_LED_vs_Qwen_Combined")


def main():
    configure_style()
    figure_4_1()
    figure_4_2()


if __name__ == "__main__":
    main()
