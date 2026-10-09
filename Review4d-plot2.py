import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. Input and output files
# ============================================================
csv_files = {
    2: Path(
        "output_Review4d/"
        "splash_statistics_width_2d.csv"
    ),
    4: Path(
        "output_Review4d/"
        "splash_statistics_width_4d.csv"
    ),
}

output_dir = Path("output_width_comparison")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load the two CSV files
# ============================================================
data_frames = []

for width, filename in csv_files.items():
    if not filename.exists():
        raise FileNotFoundError(
            f"Input file not found: {filename}"
        )

    df = pd.read_csv(filename)

    # Ensure that the width is identified correctly
    df["width_d"] = width

    data_frames.append(df)

data = pd.concat(
    data_frames,
    ignore_index=True
)


# ============================================================
# 3. Clean column values
# ============================================================
data["quantity"] = (
    data["quantity"]
    .astype(str)
    .str.strip()
)

# Allow for possible alternative spellings in the CSV files
data["quantity"] = data["quantity"].replace({
    "PR": "Pr",
    "pr": "Pr",
    "NE": "N_E",
    "N_E ": "N_E",
})

numeric_columns = [
    "width_d",
    "moisture_pct",
    "speed_normalized",
    "mean",
    "std",
    "sample_count",
]

for column in numeric_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

print("Quantities found in the CSV files:")
print(data["quantity"].unique())


# ============================================================
# 4. Retain moisture contents available for both widths
# ============================================================
moisture_2d = set(
    data.loc[
        data["width_d"] == 2,
        "moisture_pct"
    ].dropna()
)

moisture_4d = set(
    data.loc[
        data["width_d"] == 4,
        "moisture_pct"
    ].dropna()
)

common_moisture = sorted(
    moisture_2d.intersection(moisture_4d)
)

data = data[
    data["moisture_pct"].isin(common_moisture)
].copy()

print("Common moisture contents:")
print(common_moisture)


# ============================================================
# Calculate sample-count-weighted means and pooled std values
# ============================================================
summary_rows = []

group_columns = [
    "width_d",
    "moisture_pct",
    "quantity"
]

for group_values, group in data.groupby(group_columns):
    width, moisture, quantity = group_values

    means = group["mean"].to_numpy(dtype=float)
    stds = group["std"].to_numpy(dtype=float)
    counts = group["sample_count"].to_numpy(dtype=float)

    # Retain bins with a valid mean and positive sample count
    valid = (
        np.isfinite(means)
        & np.isfinite(counts)
        & (counts > 0)
    )

    means = means[valid]
    stds = stds[valid]
    counts = counts[valid]

    if means.size == 0:
        continue

    total_count = np.sum(counts)

    # Sample-count-weighted mean
    weighted_mean = np.sum(
        counts * means
    ) / total_count

    summary_rows.append({
        "width_d": width,
        "moisture_pct": moisture,
        "quantity": quantity,
        "mean": weighted_mean,
        "sample_count": total_count,
        "number_of_bins": means.size
    })


summary = pd.DataFrame(summary_rows)

summary = summary.sort_values(
    ["quantity", "width_d", "moisture_pct"]
).reset_index(drop=True)

print("\nSample-count-weighted statistics:")
print(summary)

# ============================================================
# 6. Figure settings
# ============================================================
quantities = [
    "e",
    "theta_re",
    "Pr",
    "N_E",
    "U_E",
    "theta_E",
]

ylabels = {
    "e": r"$e$ [-]",
    "theta_re": (
        r"$\theta_{\mathrm{re}}$ [$^\circ$]"
    ),
    "Pr": r"$\mathrm{Pr}$ [-]",
    "N_E": r"$N_{\mathrm{E}}$ [-]",
    "U_E": (
        r"$U_{\mathrm{E}}/\sqrt{gd}$ [-]"
    ),
    "theta_E": (
        r"$\theta_{\mathrm{E}}$ [$^\circ$]"
    ),
}

# Limits selected from the current results
ylimits = {
    "e": (0.0, 1),
    "theta_re": (10, 30),
    "Pr": (0,1),
    "N_E": (0,1),
    "U_E": (0, 15),
    "theta_E": (15, 30),
}

styles = {
    2: {
        "color": "black",
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": "black",
        "label": r"$L_y=2d$",
    },
    4: {
        "color": "tab:blue",
        "marker": "s",
        "linestyle": "--",
        "markerfacecolor": "white",
        "label": r"$L_y=4d$",
    },
}

panel_labels = [
    "(a)", "(b)", "(c)",
    "(d)", "(e)", "(f)"
]


# ============================================================
# 7. Create figure
# ============================================================
fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 7.5)
)

for ax, quantity, panel_label in zip(
    axes.flat,
    quantities,
    panel_labels
):
    quantity_data = summary[
        summary["quantity"] == quantity
    ]

    if quantity_data.empty:
        print(
            f"Warning: no data found for "
            f"quantity '{quantity}'"
        )

    for width in [2, 4]:
        subset = (
            quantity_data[
                quantity_data["width_d"] == width
            ]
            .sort_values("moisture_pct")
        )

        if subset.empty:
            print(
                f"Warning: no {width}d data found "
                f"for '{quantity}'"
            )
            continue

        x = subset[
            "moisture_pct"
        ].to_numpy(dtype=float)

        y = subset[
            "mean"
        ].to_numpy(dtype=float)

        style = styles[width]

        # Always plot the mean values
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markerfacecolor=(
                style["markerfacecolor"]
            ),
            markeredgecolor=style["color"],
            markeredgewidth=1.2,
            linewidth=1.5,
            markersize=6,
            label=style["label"],
            zorder=3,
        )

    ax.set_xlabel(
        r"$\Omega$ [$\%$]",
        fontsize=13
    )
    ax.set_ylabel(
        ylabels[quantity],
        fontsize=13
    )

    ax.set_xticks(common_moisture)
    ax.set_ylim(ylimits[quantity])

    ax.text(
        0.03,
        0.95,
        panel_label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )

    ax.tick_params(
        axis="both",
        labelsize=11
    )

    ax.grid(
        alpha=0.2,
        zorder=0
    )


# ============================================================
# 8. Shared legend and figure layout
# ============================================================
handles, labels = (
    axes[0, 0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    frameon=False,
    fontsize=12,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)

plt.show()