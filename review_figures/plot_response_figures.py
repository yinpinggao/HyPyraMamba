import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {"SSFTT": "#1769AA", "MASSFormer": "#D95F02", "PyS2CF-Mamba": "#238B45"}
PATCHES = np.array([15, 17, 21, 25, 29, 33])

patch_data = {
    "LongKou": {
        "SSFTT": {"OA": [96.15, 96.35, 95.56, 95.39, 95.67, 95.24], "std": [0.41, 0.68, 0.35, 0.46, 0.41, 1.16]},
        "MASSFormer": {"OA": [96.55, 96.82, 96.59, 96.10, 95.75, 95.94], "std": [1.59, 0.99, 1.41, 0.66, 1.07, 0.96]},
        "PyS2CF-Mamba": (98.41, 0.43),
    },
    "QUH-Qingyun": {
        "SSFTT": {"OA": [88.79, 89.72, 89.86, 90.65, 90.73, 90.57], "std": [0.82, 0.41, 0.76, 0.72, 0.60, 1.07]},
        "MASSFormer": {"OA": [88.63, 89.36, 88.42, 89.24, 89.16, 89.15], "std": [1.02, 1.10, 1.04, 0.63, 1.03, 1.13]},
        "PyS2CF-Mamba": (91.38, 1.19),
    },
    "QUH-Tangdaowan": {
        "SSFTT": {"OA": [94.79, 95.74, 95.31, 95.72, 95.18, 95.01], "std": [0.58, 0.19, 0.28, 0.08, 0.53, 0.25]},
        "MASSFormer": {"OA": [94.01, 94.69, 94.38, 94.75, 94.09, 94.93], "std": [0.54, 0.82, 0.48, 0.63, 0.61, 0.70]},
        "PyS2CF-Mamba": (96.56, 0.58),
    },
}


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_patch_sweep():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    for ax, (dataset, values) in zip(axes, patch_data.items()):
        for method in ("SSFTT", "MASSFormer"):
            d = values[method]
            ax.errorbar(
                PATCHES, d["OA"], yerr=d["std"], marker="o" if method == "SSFTT" else "s",
                linewidth=2.0, markersize=5.5, capsize=3, color=COLORS[method],
                label=method, alpha=0.95,
            )
        mean, std = values["PyS2CF-Mamba"]
        ax.axhline(mean, color=COLORS["PyS2CF-Mamba"], linewidth=2.2, label="PyS²CF-Mamba (whole image)")
        ax.fill_between(PATCHES, mean - std, mean + std, color=COLORS["PyS2CF-Mamba"], alpha=0.10, linewidth=0)
        ax.set_title(dataset)
        ax.set_xticks(PATCHES)
        ax.set_xlabel("Patch size $P$")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
        ax.set_ylim({"LongKou": (93.5, 99.2), "QUH-Qingyun": (87.0, 92.8), "QUH-Tangdaowan": (93.0, 98.0)}[dataset])
    axes[0].set_ylabel("OA (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Patch-size sensitivity of the strongest patch-based competitors", y=1.16, fontsize=14, fontweight="bold")
    save(fig, "Fig_R1_patch_size_sensitivity")


def plot_envelope():
    datasets = ["LongKou", "QUH-Qingyun", "QUH-Tangdaowan"]
    metrics = ["OA", "AA", r"$\kappa$"]
    envelope = np.array([[96.82, 96.69, 95.86], [90.73, 91.78, 87.40], [95.74, 97.73, 94.82]])
    proposed = np.array([[98.41, 98.55, 97.91], [91.38, 92.25, 88.67], [96.56, 97.94, 96.10]])
    x = np.arange(len(metrics))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), constrained_layout=True)
    for dataset_idx, (ax, dataset) in enumerate(zip(axes, datasets)):
        bars_base = ax.bar(
            x - width / 2,
            envelope[dataset_idx],
            width,
            color="#B8C6D1",
            edgecolor="#78909C",
            linewidth=0.7,
            hatch="///",
            label="Best patch envelope",
        )
        bars_prop = ax.bar(
            x + width / 2,
            proposed[dataset_idx],
            width,
            color=COLORS["PyS2CF-Mamba"],
            edgecolor="#1B5E20",
            linewidth=0.7,
            label="PyS²CF-Mamba",
        )
        for b, val in zip(bars_base, envelope[dataset_idx]):
            ax.text(b.get_x() + b.get_width()/2, val + 0.12, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#455A64")
        for b, val in zip(bars_prop, proposed[dataset_idx]):
            ax.text(b.get_x() + b.get_width()/2, val + 0.12, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#1B5E20")
        ax.set_xticks(x, metrics)
        ax.set_title(dataset)
        ax.set_ylim({"LongKou": (94.5, 99.4), "QUH-Qingyun": (85.8, 93.2), "QUH-Tangdaowan": (93.5, 99.0)}[dataset])
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    axes[0].set_ylabel("Performance (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Empirical upper envelope of tested patch configurations", y=1.15, fontsize=14, fontweight="bold")
    save(fig, "Fig_R2_patch_envelope_comparison")


def plot_context_sensitivity():
    positions = np.arange(5)
    labels = ["16", "32", "64", "128", "full"]
    data = {
        "LongKou": ([96.30, 97.24, 97.08, 98.02, 98.41], [0.53, 0.52, 0.63, 0.18, 0.43]),
        "QUH-Qingyun": ([86.60, 89.01, 89.36, 90.18, 91.38], [1.67, 1.40, 0.57, 1.73, 1.19]),
        "QUH-Tangdaowan": ([91.74, 94.13, 94.25, 96.39, 96.56], [1.08, 0.38, 0.40, 0.36, 0.58]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    for ax, (dataset, (vals, stds)) in zip(axes, data.items()):
        ax.errorbar(positions[:4], vals[:4], yerr=stds[:4], marker="o", color="#5E3C99", linewidth=2.2, markersize=5.5, capsize=3, label="Restricted blocks")
        ax.errorbar(positions[4], vals[4], yerr=stds[4], marker="*", markersize=13, color=COLORS["PyS2CF-Mamba"], capsize=3, linewidth=1.5, label="Whole-image")
        ax.annotate(f"{vals[4]:.2f}", (positions[4], vals[4]), xytext=(0, 10), textcoords="offset points", ha="center", color="#1B5E20", fontsize=8)
        ax.set_title(dataset)
        ax.set_xlabel("Spatial-support setting")
        ax.set_xticks(positions, labels)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    axes[0].set_ylabel("OA (%)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Internal spatial-context sensitivity of PyS²CF-Mamba", y=1.06, fontsize=14, fontweight="bold")
    fig.text(0.5, -0.04, "This is an internal sensitivity analysis; block size is not treated as an equivalent centered patch size.", ha="center", fontsize=9, color="#455A64")
    save(fig, "Fig_R3_pys2cf_context_sensitivity")


if __name__ == "__main__":
    plot_patch_sweep()
    plot_envelope()
    plot_context_sensitivity()
    print(f"Wrote figures to {OUT}")
