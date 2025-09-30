import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
import itertools
import matplotlib.colors as mcolors
import re
import numpy as np

STRATEGY_DISPLAY_NAMES = {
    "Centroids_saturation_high": "Cent_sat_high",
    "Centroids_saturation_medium": "Cent_sat_med",
    "Centroids_saturation_low": "Cent_sat_low",
    "Top5Similarity": "T5S",
    "Max Comp": "Max Comp",
    "Min Comp": "Min Comp",
    "Random": "Random",
    "LHS": "LHS",
    "K-Means": "K-Means",
    "Farthest": "FPS",
    "K-Center": "K-Center",
    "ODAL": "ODAL"
}
base_strategies = [
    "Top5Similarity", "Max Comp", "Min Comp", 
    "Centroids_saturation_high", "Centroids_saturation_medium", "Centroids_saturation_low",
    "Random", "LHS", "K-Means", "Farthest", "K-Center", "ODAL"
]
def plot_strategy_across_datasets(
    strategy,
    dataset_paths,
    dataset_labels,
    save_path=None,
    measurement_uncertainty=0.005,
    title=None
):
    distinct_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    iterations = list(range(100))

    for i, path in enumerate(dataset_paths):
        if not os.path.exists(path):
            print(f"[Warning] File not found: {path}")
            continue

        df = pd.read_csv(path)
        if strategy not in df.columns:
            print(f"[Warning] Strategy '{strategy}' not found in: {path}")
            continue

        raw_values = df[strategy]
        interpolated = raw_values.interpolate(limit_direction='both')
        values = interpolated.mask(raw_values.isna()).values[:100]

        color = distinct_colors[i % len(distinct_colors)]
        label = dataset_labels[i] if i < len(dataset_labels) else f"Dataset {i+1}"

        ax.plot(
            iterations[:len(values)],
            values,
            label=label,
            color=color,
            linestyle='-',
            linewidth=1.5
        )

    ax.axhline(
        y=measurement_uncertainty,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Measurement Uncertainty'
    )

    # Axis labels (larger font)
    ax.set_xlabel("Iteration", fontsize=15, labelpad=8)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=15, labelpad=8)

    # Tick labels (larger font)
    ax.tick_params(axis='both', labelsize=13)

    # Explanatory note below x-label
    ax.text(
        0.5, -0.18,
        "Total number of measurements = Iteration + 5 (Initial measurements)",
        transform=ax.transAxes,
        fontsize=11,
        ha='center',
        va='top'
    )

    plt.subplots_adjust(bottom=0.18)

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([0, 100])

    if title:
        ax.set_title(title, fontsize=16, pad=14)

    # Legend with bigger font
    ax.legend(
        title="Dataset",
        fontsize=12,          # legend entries
        title_fontsize=13,    # legend title
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=True,
        handletextpad=0.4,
        columnspacing=1.2
    )

    plt.tight_layout()

    if save_path:
        if not save_path.lower().endswith(".pdf"):
            save_path = save_path.rsplit('.', 1)[0] + ".pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()


def get_large_color_palette(n):
    """Return n distinct colors by combining colormaps."""
    cmap_list = ['tab10', 'tab20', 'tab20b', 'tab20c', 'Set3', 'Paired', 'Pastel1']
    color_list = []

    for cmap_name in cmap_list:
        cmap = cm.get_cmap(cmap_name)
        for i in range(cmap.N):
            rgba = cmap(i)
            color_list.append(mcolors.to_hex(rgba))
            if len(color_list) >= n:
                return color_list
    return color_list[:n]


def plot_all_base_and_mixed_strategies(df, main_strategy, base_strategies, save_path=None, measurement_uncertainty=0.005):
    strategies_to_plot = []
    labels = []
    styles = []

    full_strategy_list = []

    for base in base_strategies:
        full_strategy_list.append(base)
        full_strategy_list.append(f"{main_strategy}+{base}")

    color_palette = get_large_color_palette(len(full_strategy_list))
    color_map = dict(zip(full_strategy_list, color_palette))

    for base in base_strategies:
        base_label = STRATEGY_DISPLAY_NAMES.get(base, base)
        mixed_label = f"{STRATEGY_DISPLAY_NAMES.get(main_strategy, main_strategy)}+{base_label}"

        if base in df.columns:
            strategies_to_plot.append(base)
            labels.append(base_label)
            styles.append(("solid", color_map[base]))

        mixed_name = f"{main_strategy}+{base}"
        if mixed_name in df.columns:
            strategies_to_plot.append(mixed_name)
            labels.append(mixed_label)
            styles.append(("dashed", color_map[mixed_name]))

    if not strategies_to_plot:
        print("No matching strategies found in the data.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    iterations = list(range(100))

    for strategy, label, (linestyle, color) in zip(strategies_to_plot, labels, styles):
        raw_values = df[strategy] if strategy in df.columns else pd.Series([None]*100)
        interpolated = raw_values.interpolate(limit_direction='both')
        values = interpolated.mask(raw_values.isna()).values[:100]

        ax.plot(
            iterations[:len(values)],
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2
        )

    ax.axhline(
        y=measurement_uncertainty,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label='Measurement Uncertainty'
    )

    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim([0, 100])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add explanation below the plot
    plt.xlabel("Iteration", labelpad=5)  # Decrease padding from axis to label
    plt.text(0.5, -0.12,
            "Total number of measurements = Iteration + 10 (Initial measurements)",
            fontsize=10,
            ha='center',
            va='top',
            transform=plt.gca().transAxes)



    ax.legend(
        title="Strategy",
        fontsize=12,
        title_fontsize=14,
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=True,
        handletextpad=0.4,
        columnspacing=1.2
    )

    plt.tight_layout()

    if save_path:
        if not save_path.endswith(".pdf"):
            save_path = save_path.rsplit('.', 1)[0] + ".pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()



def plot_average_mae_across_seeds(base_dir, main_strategy, base_strategies, save_path=None, measurement_uncertainty=0.005):
    """
    Reads mae_priors_results.csv from all seed subfolders in base_dir,
    averages MAE across seeds for each (base) and (main+base),
    then plots them like plot_all_base_and_mixed_strategies.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import re

    # --- Collect all seed files ---
    seed_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    all_dfs = []
    
    for sd in seed_dirs:
        csv_path = os.path.join(base_dir, sd, "mae_priors_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
        else:
            print(f" No mae_priors_results.csv in {sd}, skipping.")

    if not all_dfs:
        print("No MAE data found.")
        return

    # --- Combine all into one big DF ---
    combined = pd.concat(all_dfs, axis=1)  # columns from all seeds side by side

    # --- Normalize column names by stripping _seed_x ---
    new_cols = []
    for c in combined.columns:
        base_name = re.sub(r'_seed_\d+$', '', c)
        new_cols.append(base_name)
    combined.columns = new_cols

    # --- Average across all columns with same base_name ---
    avg_df = combined.T.groupby(level=0).mean().T

    # --- Build strategy list like original function ---
    strategies_to_plot = []
    labels = []
    styles = []

    full_strategy_list = []
    for base in base_strategies:
        full_strategy_list.append(base)
        full_strategy_list.append(f"{main_strategy}+{base}")

    # Define colors
    color_palette = get_large_color_palette(len(full_strategy_list))
    color_map = dict(zip(full_strategy_list, color_palette))

    for base in base_strategies:
        base_label = STRATEGY_DISPLAY_NAMES.get(base, base)
        mixed_label = f"{STRATEGY_DISPLAY_NAMES.get(main_strategy, main_strategy)}+{base_label}"

        if base in avg_df.columns:
            strategies_to_plot.append(base)
            labels.append(base_label)
            styles.append(("solid", color_map[base]))

        mixed_name = f"{main_strategy}+{base}"
        if mixed_name in avg_df.columns:
            strategies_to_plot.append(mixed_name)
            labels.append(mixed_label)
            styles.append(("dashed", color_map[mixed_name]))

    if not strategies_to_plot:
        print("No matching strategies found in the averaged data.")
        return

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    iterations = list(range(len(avg_df)))

    for strategy, label, (linestyle, color) in zip(strategies_to_plot, labels, styles):
        raw_values = avg_df[strategy] if strategy in avg_df.columns else pd.Series([None]*len(iterations))
        interpolated = raw_values.interpolate(limit_direction='both')
        values = interpolated.mask(raw_values.isna()).values

        ax.plot(
            iterations[:len(values)],
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2
        )

    ax.axhline(
        y=measurement_uncertainty,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label='Measurement Uncertainty'
    )

    ax.set_xlabel("Iteration", fontsize=14, labelpad=5)  # Custom label with reduced padding
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([0, len(iterations)-1])

    # Add custom note under the x-axis
    ax.text(
        0.5, -0.12,
        "Total number of measurements = Iteration + 10 (Initial measurements)",
        fontsize=10,
        ha='center',
        va='top',
        transform=ax.transAxes
    )

    ax.legend(
        title="Strategy",
        fontsize=12,
        title_fontsize=14,
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        frameon=True,
        handletextpad=0.4,
        columnspacing=1.2
    )

    plt.tight_layout()

    if save_path:
        if not save_path.endswith(".pdf"):
            save_path = save_path.rsplit('.', 1)[0] + ".pdf"
        plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
def plot_initialization_strategies(csv_path, all_init_strategies,
                                   strategy_order=None,
                                   resistance_col="Resistance", x_col="x", y_col="y",
                                   max_points=342, output_path=None):
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.read_csv(csv_path).iloc[:max_points]
    x = data[x_col].values
    y = data[y_col].values
    resistance = data[resistance_col].values

    cmap = plt.colormaps["plasma"]

    # Use given order or default to dict order
    if strategy_order is None:
        strategy_order = list(all_init_strategies.keys())

    num_strategies = len(strategy_order)
    cols = 4
    rows = (num_strategies // cols) + (num_strategies % cols > 0)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    vmin, vmax = resistance.min(), resistance.max()
    scatter = None  # For the colorbar

    for idx, strategy in enumerate(strategy_order):
        if idx >= len(axes):
            break

        indices = all_init_strategies[strategy]
        ax = axes[idx]
        ax.set_aspect("equal")

        scatter = ax.scatter(x, y, c=resistance, cmap=cmap, marker="s", s=50, vmin=vmin, vmax=vmax)
        ax.scatter(x[indices], y[indices], c="white", marker="X", s=200, edgecolor="black", linewidth=2)
        ax.scatter(x[indices], y[indices], c="red", marker="o", s=100, edgecolor="black", linewidth=1, alpha=0.8)

        for i in indices:
            ax.text(x[i], y[i], 'X', fontsize=12, color='black', ha='center', va='center', fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(strategy, fontsize=14)

    # Hide unused subplots
    for i in range(idx + 1, len(axes)):
        axes[i].axis("off")

    # Add colorbar inside the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Resistance (Ohm)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar

    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches='tight', dpi=300)
    else:
        plt.show()
