import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import pandas as pd

import pandas as pd
import numpy as np


def prepare_radar_data(df, acc_metric, beyond_metrics):
    """
    Prepara i dati per il Kiwiat Plot (Spider Plot).

    1. Ordina il DataFrame.
    2. Calcola Min/Max globali per una scala coerente tra i modelli.
    3. Inverte le metriche di 'bias' (es. Gini, PopREO) in modo che 'esterno' = 'migliore'.
    4. Restituisce liste sincronizzate di dati ed etichette.
    """

    # 1. Ordinamento fondamentale per evitare discrepanze nei grafici
    # Ordiniamo per Algoritmo e poi per numero di vicini (nn) crescente
    df = df.sort_values(by=['Algorithm', 'nn']).reset_index(drop=True)

    # Uniamo le metriche in un'unica lista
    all_metrics = [acc_metric] + list(beyond_metrics)

    # Identifichiamo i primi due algoritmi presenti (es. ItemkNN e UserkNN)
    algorithms = df['Algorithm'].unique()[:2]

    data_list = []
    labels_list = []

    # 2. Calcolo Min-Max GLOBALE
    # È cruciale calcolarlo su TUTTI i dati insieme, altrimenti un modello scarso
    # sembrerebbe eccellente se normalizzato solo su se stesso.
    raw_all = df[all_metrics].values.astype(float)
    min_v = raw_all.min(axis=0)
    max_v = raw_all.max(axis=0)

    # Evitiamo la divisione per zero se max == min
    range_v = np.where((max_v - min_v) == 0, 1, max_v - min_v)

    # 3. Iterazione per algoritmo (creazione dei due subplot)
    for algo in algorithms:
        # Filtriamo il dataframe per l'algoritmo corrente
        df_algo = df[df['Algorithm'] == algo].copy()


        # A. Estrazione e Normalizzazione Dati
        raw_values = df_algo[all_metrics].values.astype(float)
        norm_values = (raw_values - min_v) / range_v

        # B. Inversione metriche di Bias (Lower is Better -> Higher is Better)
        # Se la metrica è di disparità la invertiamo (1 - x)
        # così che un punto esterno sul radar significhi sempre "Performance Migliore"
        bias_keywords = ['popreo', 'reo', 'rsp', 'bias']
        for i, m_name in enumerate(all_metrics):
            if any(bias in m_name.lower() for bias in bias_keywords):
                norm_values[:, i] = 1 - norm_values[:, i]

        # C. Creazione Etichette (Labels)
        # Generandole qui, siamo sicuri al 100% che il numero di label
        # corrisponda esattamente al numero di righe di dati (evita IndexError)
        labels = [f"nn={int(n)}" for n in df_algo['nn']]

        data_list.append(norm_values)
        labels_list = labels

    return data_list, labels_list, all_metrics



# Function to create a spider plot
def create_spider_plot(data, models, metrics, titles, num_plots=2):
    num_vars = len(metrics)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Initialize plot
    fig, axs = plt.subplots(1, num_plots, figsize=(8, 5), subplot_kw=dict(polar=True))

    for i, ax in enumerate(axs):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable and add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, color='black', size=11)

        # Remove y-ticks
        ax.set_yticklabels([])

        # Remove the outermost circle
        ax.spines['polar'].set_visible(False)

        # Draw background
        ax.fill(np.linspace(0, 2 * np.pi, 100), np.ones(100), color='lightgrey', alpha=0.5)

        # Plot each model
        current_plot_data = data[i]
        current_labels = models[i]
        for j, model_data in enumerate(current_plot_data):
            values = model_data.tolist()
            values += values[:1]

            line, = ax.plot(angles, values, linewidth=2, label=current_labels[j])
            ax.fill(angles, values, color=line.get_color(), alpha=0.25)
            ax.scatter(angles[:-1], model_data, s=20, color=line.get_color(), zorder=3)

        # Titolo posizionato in alto per non sovrapporsi
        ax.set_title(titles[i], fontsize=15, pad=50, y=1.1, fontweight='bold')

        # Legenda posizionata sotto ogni singolo grafico
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)

    # Add a legend below the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=5, fontsize=12)
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_labels), fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)


    # Show the plot
    # plt.show()
    return fig
