import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import argparse


def load_data(data_dir, metrics, neighbors):
    """
    Carica i dati per N metriche, gestendo 'ApproxSeverity'.
    """
    model_data = {}
    csv_metrics = [m for m in metrics if m != "ApproxSeverity"]

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} non trovata.")
        return {}

    for filename in os.listdir(data_dir):
        if filename.startswith("target_t_") and filename.endswith(".csv"):
            try:
                t_str = filename.replace("target_t_", "").replace(".csv", "").replace("_minh", "")
                t_val = float(t_str.replace("_", "."))
            except ValueError:
                continue

            filepath = os.path.join(data_dir, filename)
            is_minh_file = "_minh" in filename

            try:
                df = pd.read_csv(filepath)
                if 'nn' not in df.columns or any(m not in df.columns for m in csv_metrics):
                    continue

                filtered_df = df[df['nn'] == neighbors].copy()

                if not filtered_df.empty:
                    for index, row in filtered_df.iterrows():
                        algo = row.get('Algorithm', 'Unknown')
                        sim = row.get('sim', 'Unknown')
                        strat = str(row.get('strat', '')).strip() if pd.notna(row.get('strat')) else ''
                        algo_str = str(algo)
                        sim_str = str(sim).lower()

                        if not is_minh_file:
                            if 'FairANN' in algo_str and (
                                    strat == '' or strat.lower() == 'no_sampling') and sim_str == 'jaccard':
                                continue

                        if 'FairANN' in algo_str and (strat == '' or strat.lower() == 'no_sampling'):
                            prefix = 'User' if algo_str.startswith('User') else 'Item'
                            if sim_str == 'jaccard':
                                label = f"{prefix}Minhashing"
                            elif sim_str == 'cosine':
                                label = f"{prefix}LSHRandomProjection"
                            else:
                                label = f"{algo} - {sim}"
                        else:
                            parts = [str(algo), str(sim)]
                            if strat: parts.append(strat)
                            label = " - ".join(parts)

                        model_type = 'User' if algo_str.startswith('User') else 'Item'

                        if label not in model_data:
                            model_data[label] = {
                                'type': model_type, 'sim': str(sim), 'algo': str(algo)
                            }
                            for m in metrics:
                                model_data[label][m] = []

                        for m in metrics:
                            val = t_val if m == "ApproxSeverity" else row[m]
                            model_data[label][m].append(val)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return model_data


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_2d(model_data, x_metric, y_metric, neighbors, title_suffix="", output_file=None):
    if not model_data: return
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, (label, data) in enumerate(model_data.items()):
        combined = sorted(zip(data[x_metric], data[y_metric]), key=lambda k: k[0])
        x_vals, y_vals = zip(*combined) if combined else ([], [])
        plt.scatter(x_vals, y_vals, label=label, marker=markers[i % len(markers)], alpha=0.8, s=60)
        plt.plot(x_vals, y_vals, alpha=0.4, linewidth=1.5)

    plt.title(f'{x_metric} vs {y_metric} (nn={neighbors}) {title_suffix}')
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
    plt.close()


def plot_3d(model_data, x_metric, y_metric, z_metric, neighbors, title_suffix="", output_file=None):
    if not model_data: return
    fig = go.Figure()
    for label, data in model_data.items():
        combined = sorted(zip(data[x_metric], data[y_metric], data[z_metric]), key=lambda k: k[0])
        x_v, y_v, z_v = zip(*combined) if combined else ([], [], [])
        fig.add_trace(go.Scatter3d(
            x=x_v, y=y_v, z=z_v, mode='lines+markers', name=label,
            marker=dict(size=4), line=dict(width=4),
            text=[f"{label}<br>{x_metric}:{x:.3f}<br>{y_metric}:{y:.3f}<br>{z_metric}:{z:.3f}" for x, y, z in
                  zip(x_v, y_v, z_v)]
        ))
    fig.update_layout(
        title=f'3D: {x_metric} vs {y_metric} vs {z_metric} (nn={neighbors}) {title_suffix}',
        scene=dict(xaxis_title=x_metric, yaxis_title=y_metric, zaxis_title=z_metric),
        margin=dict(l=0, r=0, b=0, t=50), legend=dict(x=0, y=1)
    )
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if not output_file.endswith(".html"): output_file += ".html"
        fig.write_html(output_file)


# ==============================================================================
# MAIN AUTOMATION
# ==============================================================================

def run_automation(data_dir, metrics, split_sim, split_type):
    neighbor_values = [50, 100, 250]
    is_3d = (len(metrics) == 3)

    # Nome cartella principale basato sulle metriche (Esempio: nDCG_vs_CR_vs_ApproxSeverity)
    if is_3d:
        main_folder = f"{metrics[0]}_vs_{metrics[1]}_vs_{metrics[2]}"
    else:
        main_folder = f"{metrics[0]}_vs_{metrics[1]}"

    print(f"Inizio generazione batch per: {main_folder}")

    for nn in neighbor_values:
        print(f"  > Processing nn={nn}")
        model_data = load_data(data_dir, metrics, nn)

        if not model_data:
            continue

        # Percorso: plots / METRICHE / nn_X /
        base_out_dir = os.path.join("plots", main_folder, f"nn_{nn}")

        # Raggruppamento
        if split_sim:
            groups = {
                "User_Cosine": {k: v for k, v in model_data.items() if
                                v['type'] == 'User' and v['sim'].lower() in ['cosine', 'angular']},
                "User_Jaccard": {k: v for k, v in model_data.items() if
                                 v['type'] == 'User' and v['sim'].lower() == 'jaccard'},
                "Item_Cosine": {k: v for k, v in model_data.items() if
                                v['type'] == 'Item' and v['sim'].lower() in ['cosine', 'angular']},
                "Item_Jaccard": {k: v for k, v in model_data.items() if
                                 v['type'] == 'Item' and v['sim'].lower() == 'jaccard'}
            }
        elif split_type:
            groups = {
                "User": {k: v for k, v in model_data.items() if v['type'] == 'User'},
                "Item": {k: v for k, v in model_data.items() if v['type'] == 'Item'}
            }
        else:
            groups = {"All_Models": model_data}

        for g_name, g_data in groups.items():
            if not g_data: continue
            ext = ".html" if is_3d else ".png"
            out_path = os.path.join(base_out_dir, f"{g_name}{ext}")

            if is_3d:
                plot_3d(g_data, metrics[0], metrics[1], metrics[2], nn, title_suffix=f"({g_name})",
                        output_file=out_path)
            else:
                plot_2d(g_data, metrics[0], metrics[1], nn, title_suffix=f"({g_name})", output_file=out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--metrics", nargs='+', required=True, help="2 o 3 metriche (X Y [Z])")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--split_sim", action="store_true")
    args = parser.parse_args()

    if len(args.metrics) not in [2, 3]:
        print("Errore: specifica 2 o 3 metriche.")
    else:
        run_automation(args.data_dir, args.metrics, args.split_sim, args.split)
        print("\nBatch completato. Controlla la cartella 'plots'.")