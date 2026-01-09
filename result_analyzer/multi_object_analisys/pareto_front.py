import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import itertools


def identify_pareto(df, x_col, y_col):
    """Returns a boolean array of points on the Pareto frontier (maximizing both)."""
    data = df[[x_col, y_col]].values
    is_efficient = np.ones(data.shape[0], dtype=bool)
    for i, c in enumerate(data):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(data[is_efficient] >= c, axis=1)
            is_efficient[i] = True
    return is_efficient


def automated_mobj_report(
    df,
    plot_dir,
    acc_metric="Recall",
    beyond_metrics=["ItemCoverage", "EPC"],
    save=False,
):
    # 1. Calculate Statistical Coefficients (Spearman Rho)
    stats = []
    for model in df["Algorithm"].unique():
        model_df = df[df["Algorithm"] == model]
        for b_metric in beyond_metrics:
            rho = model_df[acc_metric].corr(model_df[b_metric], method="spearman")
            stats.append(
                {
                    "Algorithm": model,
                    "Metric_Pair": f"{acc_metric} vs {b_metric}",
                    "Spearman_Rho": rho,
                }
            )

    stats_df = pd.DataFrame(stats)
    print("--- Statistical Trade-off Summary ---")
    print(stats_df)

    # 2. Visualization with Pareto Frontiers
    plt.figure(figsize=(14, 6))
    for i, b_metric in enumerate(beyond_metrics, 1):
        plt.subplot(1, len(beyond_metrics), i)
        sns.scatterplot(
            data=df, x=acc_metric, y=b_metric, hue="Algorithm", style="Cutoff", s=100
        )

        for algo in df["Algorithm"].unique():
            algo_data = df[df["Algorithm"] == algo].copy()
            pareto_mask = identify_pareto(algo_data, acc_metric, b_metric)
            pareto_points = algo_data[pareto_mask].sort_values(acc_metric)
            plt.plot(
                pareto_points[acc_metric], pareto_points[b_metric], "--", alpha=0.5
            )

        plt.title(f"{acc_metric} vs {b_metric}")
        plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save:
        # Save filename includes the metrics for clarity
        acc_subfolder = plot_dir / acc_metric
        acc_subfolder.mkdir(parents=True, exist_ok=True)

        # Nome file basato sulle metriche beyond incluse nel grafico
        metrics_str = "_".join(beyond_metrics)
        file_name = f"{acc_metric}_vs_{metrics_str}.png"

        save_path = acc_subfolder / file_name
        plt.savefig(save_path, dpi=300)
        print(f"✅ Grafico salvato in: {save_path.absolute()}")
    else:
        plt.show()

    plt.close()  # Clean up memory
    return stats_df


# --- Main Execution ---
def mobj_pipeline(csv_path, accuracy_metrics, beyond_accuracy, save=False):

    # Create 'plots' folder in the parent of the performance folder
    plot_dir = csv_path.parent.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Generate all unique couples of beyond accuracy metrics and convert in list form
    unique_couples_list = [
        list(couple) for couple in itertools.combinations(beyond_accuracy, 2)
    ]

    #  Load and Run
    if csv_path.exists():
        df_final = pd.read_csv(csv_path)

        # Run analysis and save results
        for accuracy in accuracy_metrics:
            for beyond_pair in unique_couples_list:
                print(
                    f"Analyzing {accuracy} with {beyond_pair[0]} and {beyond_pair[1]}"
                )
                stats_results = automated_mobj_report(
                    df_final,
                    plot_dir=plot_dir,
                    acc_metric=accuracy,
                    beyond_metrics=beyond_pair,
                    save=save,
                )
    else:
        print(f"File not found: {csv_path}")
