import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import itertools

from result_analyzer.multi_object_analisys.kiwiat_plot import prepare_radar_data
from result_analyzer.multi_object_analisys.kiwiat_plot import create_spider_plot


def reset_directory(path):
    """Elimina la cartella se esiste e la ricrea vuota."""
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def identify_pareto(df, x_col, y_col):
    """Returns a boolean array of points on the Pareto frontier (maximizing both)."""
    data = df[[x_col, y_col]].values
    is_efficient = np.ones(data.shape[0], dtype=bool)

    # List of bias metrics (Lower is better)
    bias_metrics = ['popreo', 'poprsp', 'arp', 'aclt', 'aplt']

    # Check if the metrics are bias metrics
    max_x = x_col.lower() not in bias_metrics
    max_y = y_col.lower() not in bias_metrics

    # If any one of the metric is a bias, metrics, the values are inverted in sign
    if not max_x:
        data[:, 0] = -data[:, 0]
    if not max_y:
        data[:, 1] = -data[:, 1]


    for i, c in enumerate(data):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(data[is_efficient] >= c, axis=1)
            is_efficient[i] = True
    return is_efficient


def automated_mobj_report(
    df,
    plot_dir,
    kiwiat_plot_dir,
    acc_metric="Recall",
    beyond_metrics=["ItemCoverage", "EPC"],
    save=True,
    cutoff_for_radar=10
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

        for idx, row in df.iterrows():
            # Verifica che nn non sia NaN
            if pd.notna(row['nn']):
                plt.annotate(
                    text="nn="+str(int(row['nn'])),  # Il testo (numero nn)
                    xy=(row[acc_metric], row[b_metric]),  # La posizione del punto (pallino)
                    xytext=(5, 5),  # L'offset: (x, y) in punti. Qui sposta 5pt a destra e 5pt in alto
                    textcoords='offset points',  # Importante: dice che (5,5) sono punti fisici, non coordinate dati
                    fontsize=9,
                    alpha=0.9,
                    color='black'
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

    # 3. Integrazione Radar Chart (Kiwiat Plot)
    df_radar_input = df[df["Cutoff"] == cutoff_for_radar].copy()
    unique_algos = df["Algorithm"].unique()[:2]  # Prende i primi due (es. ItemkNN e UserkNN)

    all_data_for_radar = []
    all_labels_for_radar = []
    radar_titles = []

    for algo in unique_algos:
        # Filtriamo per algoritmo e cutoff fisso
        df_algo = df[(df["Algorithm"] == algo) & (df["Cutoff"] == cutoff_for_radar)].copy()

        if not df_algo.empty:
            # Prepariamo i dati (normalizzazione e inversione)
            data_list, model_names, metrics_labels = prepare_radar_data(
                df_algo, acc_metric, beyond_metrics[:2]
            )

            # data_list[0] contiene la matrice di tutti i 'nn' per questo algoritmo
            all_data_for_radar.append(data_list[0])
            all_labels_for_radar.append(model_names)
            radar_titles.append(f"Model: {algo}")

    # 2. Chiamata alla funzione con num_plots=2
    if len(all_data_for_radar) == 2:
        create_spider_plot(
            data=all_data_for_radar,
            models=all_labels_for_radar,
            metrics=metrics_labels,
            titles=radar_titles,
            num_plots=len(all_data_for_radar)
        )

        if save:
            acc_subfolder = Path(kiwiat_plot_dir) / acc_metric
            acc_subfolder.mkdir(parents=True, exist_ok=True)

            # Salvataggio con nome file descrittivo
            metrics_slug = "_".join(metrics_labels)
            radar_path = acc_subfolder / f"Radar_{metrics_slug}.png"

            # Salviamo l'ultima figura aperta (il radar)
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            print(f"✅ Radar Chart salvato in: {radar_path.absolute()}")

        plt.close()  # Chiude la figura del radar


    return stats_df


# --- Main Execution ---
def mobj_pipeline(csv_path, accuracy_metrics, beyond_accuracy, save=False):

    # Create 'plots' folder in the parent of the performance folder
    plot_dir = csv_path.parent.parent / "plots"
    kiwiat_plot_dir = csv_path.parent.parent / "kiwiat_plots"

    # Create the plot directories or reset them if already existing
    reset_directory(plot_dir)
    reset_directory(kiwiat_plot_dir)


    # Generate all unique couples of beyond accuracy metrics and convert in list form
    unique_couples_list = [
        list(couple) for couple in itertools.combinations(beyond_accuracy, 2)
    ]

    #  Load and Run
    if csv_path.exists():
        df_final = pd.read_csv(csv_path)

        # Ordina per 'Algorithm' (opzionale) e poi per 'nn' in modo crescente
        df_final = df_final.sort_values(by=['Algorithm', 'nn'], ascending=[True, True])

        # Reset dell'indice per evitare problemi con i vecchi indici rimescolati
        df_final = df_final.reset_index(drop=True)

        # Run analysis and save results
        for accuracy in accuracy_metrics:
            for beyond_pair in unique_couples_list:
                print(
                    f"Analyzing {accuracy} with {beyond_pair[0]} and {beyond_pair[1]}"
                )
                stats_results = automated_mobj_report(
                    df_final,
                    plot_dir=plot_dir,
                    kiwiat_plot_dir=kiwiat_plot_dir,
                    acc_metric=accuracy,
                    beyond_metrics=beyond_pair,
                    save=save,
                    cutoff_for_radar=10
                )
    else:
        print(f"File not found: {csv_path}")
