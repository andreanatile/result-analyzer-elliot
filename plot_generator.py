import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import argparse


def load_data(data_dir, metrics, neighbors):
    """
    Carica i dati per N metriche, gestendo 'ApproxSeverity' e i dati di signficatività da 'filtered_t_test'.
    """
    import re
    
    model_data = {}
    csv_metrics = [m for m in metrics if m != "ApproxSeverity"]

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} non trovata.")
        return {}
        
    # --- 1. Load Primary Performance Data ---
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
                        algo_orig = row.get('Algorithm', 'Unknown')
                        sim_orig = row.get('sim', 'Unknown')
                        strat_orig = str(row.get('strat', '')).strip() if pd.notna(row.get('strat')) else ''
                        preposp_orig = str(row.get('preposp', '')).strip() if pd.notna(row.get('preposp')) else ''
                        
                        algo_str = str(algo_orig)
                        strat = strat_orig
                        if 'knnfairness' in algo_str.lower() and preposp_orig:
                            strat = preposp_orig

                        sim_str = str(sim_orig).lower()
                        if sim_str == 'angular': 
                            sim_str = 'cosine'

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
                                label = f"{algo_str} - {sim_str}"
                        else:
                            parts = [str(algo_str), str(sim_str)]
                            if strat: parts.append(strat)
                            label = " - ".join(parts)

                        model_type = 'User' if algo_str.startswith('User') else 'Item'

                        if label not in model_data:
                            model_data[label] = {
                                'type': model_type, 'sim': sim_str, 'algo': algo_str
                            }
                            # Lists for metric values
                            for m in metrics:
                                model_data[label][m] = []
                            # Lists for significance boolean masks (one per metric)
                            for m in metrics:
                                model_data[label][m + "_sig"] = []

                        for m in metrics:
                            val = t_val if m == "ApproxSeverity" else row[m]
                            model_data[label][m].append(val)
                            model_data[label][m + "_sig"].append(False) # Default to false for now
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    # --- 2. Load T-Test Significance Data ---
    ttest_dir = os.path.join(os.path.dirname(data_dir), "filtered_t_test")
    if os.path.exists(ttest_dir):
        for filename in os.listdir(ttest_dir):
            if filename.endswith(".csv"):
                try:
                    match = re.search(r'(?:t_test)_?(\d+(?:_|\.)\d+)', filename, re.IGNORECASE)
                    if not match: continue
                    t_val = float(match.group(1).replace('_', '.'))
                except ValueError:
                    continue
                    
                filepath = os.path.join(ttest_dir, filename)
                try:
                    tt_df = pd.read_csv(filepath)
                    if tt_df.empty: continue
                    
                    for index, row in tt_df.iterrows():
                        if not row.get('Significant_Difference', False):
                            continue # Skip non-significant entries
                            
                        metric_name = row.get('Metric')
                        if metric_name not in csv_metrics:
                            continue
                            
                        model_str = str(row['Model_A'])
                        
                        algo_orig = model_str.split('_')[0]
                        
                        sim_match = re.search(r'sim=([a-zA-Z]+)', model_str)
                        sim_str = sim_match.group(1).lower() if sim_match else 'unknown'
                        if sim_str == 'angular': sim_str = 'cosine'

                        strat_match = re.search(r'samp_strat=([a-zA-Z_]+)(?:_val|_|$)', model_str)
                        strat = strat_match.group(1) if strat_match else ''
                        
                        preposp_match = re.search(r'preposp=([a-zA-Z-]+)(?:_|$)', model_str)
                        preposp = preposp_match.group(1) if preposp_match else ''
                        
                        if 'knnfairness' in algo_orig.lower() and preposp:
                            strat = preposp
                            
                        algo_str = algo_orig

                        # Rebuild identical label logic
                        if 'FairANN' in algo_str and (strat == '' or strat.lower() == 'no_sampling'):
                            prefix = 'User' if algo_str.startswith('User') else 'Item'
                            if sim_str == 'jaccard':
                                label = f"{prefix}Minhashing"
                            elif sim_str == 'cosine':
                                label = f"{prefix}LSHRandomProjection"
                            else:
                                label = f"{algo_str} - {sim_str}"
                        else:
                            parts = [str(algo_str), str(sim_str)]
                            if strat: parts.append(strat)
                            label = " - ".join(parts)
                            
                        # Apply True flag to the matching label + metric + index
                        if label in model_data:
                            # We must match the significance to the EXACT correct Approximate/Threshold slice.
                            # We know Model_A represents this specific `t_val`.
                            # Find the index of `t_val` in `ApproxSeverity` or directly match to values lists.
                            if "ApproxSeverity" in metrics:
                                # Safe assumption: if there's multiple runs, we map by ApproxSeverity (target_t)
                                for i, severity in enumerate(model_data[label]["ApproxSeverity"]):
                                    if abs(severity - t_val) < 0.0001:
                                        model_data[label][metric_name + "_sig"][i] = True
                            else:
                                # If ApproxSeverity isn't loaded but we need it to match the rows...
                                # This generally won't happen for plot_generator which plots curves exactly via ApproxSeverity or similar.
                                # But if it does, we assume the list indices are synced.
                                pass

                except Exception as e:
                    print(f"Error reading t-test file {filename}: {e}")
                    
    return model_data


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def get_base_label(label):
    base = label
    if base.startswith('User'): base = base[4:]
    if base.startswith('Item'): base = base[4:]
    base = base.replace(' - cosine', '').replace(' - jaccard', '').replace(' - angular', '')
    return base

def plot_2d(model_data, x_metric, y_metric, neighbors, all_labels=None, title_suffix="", output_file=None):
    if not model_data: return
    plt.figure(figsize=(12, 8))
    
    if all_labels is None:
        all_labels = list(model_data.keys())
        
    base_labels = sorted(set(get_base_label(l) for l in all_labels))
    colors = plt.cm.tab10.colors
    if len(base_labels) > 10:
        colors = plt.cm.tab20.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', 'd']

    for label, data in model_data.items():
        base = get_base_label(label)
        idx = base_labels.index(base)
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        
        # We need to zip x, y, x_sig, y_sig so they all sort together
        x_sig_key, y_sig_key = x_metric + "_sig", y_metric + "_sig"
        combined = sorted(zip(data[x_metric], data[y_metric], data[x_sig_key], data[y_sig_key]), key=lambda k: k[0])
        
        if combined:
            x_vals, y_vals, x_sigs, y_sigs = zip(*combined)
            
            # Base line and markers
            plt.scatter(x_vals, y_vals, label=label, marker=m, color=c, alpha=0.8, s=60)
            plt.plot(x_vals, y_vals, color=c, alpha=0.4, linewidth=1.5)
            
            # Highlight significant points with a star
            sig_x = [xv for xv, yv, xs, ys in combined if xs or ys]
            sig_y = [yv for xv, yv, xs, ys in combined if xs or ys]
            if sig_x:
                x_range = max(max(x_vals) - min(x_vals), 0.001)
                y_range = max(max(y_vals) - min(y_vals), 0.001)
                offset_x = x_range * 0.02
                offset_y = y_range * 0.1
                
                sig_x_shifted = [x + offset_x for x in sig_x]
                sig_y_shifted = [y + offset_y for y in sig_y]
                
                plt.scatter(sig_x_shifted, sig_y_shifted, color='black', marker='$*$', s=10, alpha=0.4, zorder=5, label='_nolegend_')

    plt.title(f'{x_metric} vs {y_metric} (nn={neighbors}) {title_suffix}\n(* = significatività stat.)')
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
    plt.close()


def plot_2d_combined(group_data_by_nn, x_metric, y_metric, neighbor_values, all_labels=None, title_suffix="", output_file=None):
    if not any(group_data_by_nn.values()): return
    
    fig, axes = plt.subplots(1, len(neighbor_values), figsize=(6 * len(neighbor_values), 8))
    if len(neighbor_values) == 1:
        axes = [axes]
        
    if all_labels is None:
        all_labels = set()
        for nn, m_data in group_data_by_nn.items():
            all_labels.update(m_data.keys())
        all_labels = list(all_labels)
        
    base_labels = sorted(set(get_base_label(l) for l in all_labels))
    colors = plt.cm.tab10.colors
    if len(base_labels) > 10:
        colors = plt.cm.tab20.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', 'd']

    for i, nn in enumerate(neighbor_values):
        ax = axes[i]
        model_data = group_data_by_nn.get(nn, {})
        for label, data in model_data.items():
            base = get_base_label(label)
            idx = base_labels.index(base)
            c = colors[idx % len(colors)]
            m = markers[idx % len(markers)]
            
            x_sig_key, y_sig_key = x_metric + "_sig", y_metric + "_sig"
            combined = sorted(zip(data[x_metric], data[y_metric], data[x_sig_key], data[y_sig_key]), key=lambda k: k[0])
            
            if combined:
                x_vals, y_vals, x_sigs, y_sigs = zip(*combined)
                
                # Base line and markers
                ax.scatter(x_vals, y_vals, label=label if i == 0 else "", marker=m, color=c, alpha=0.8, s=60)
                ax.plot(x_vals, y_vals, color=c, alpha=0.4, linewidth=1.5)
                # Highlight significant points with an exact asterisk text marker
                sig_x = [xv for xv, yv, xs, ys in combined if xs or ys]
                sig_y = [yv for xv, yv, xs, ys in combined if xs or ys]
                if sig_x:
                    x_range = max(max(x_vals) - min(x_vals), 0.001)
                    y_range = max(max(y_vals) - min(y_vals), 0.001)
                    offset_x = x_range * 0.02
                    offset_y = y_range * 0.1
                    
                    sig_x_shifted = [x + offset_x for x in sig_x]
                    sig_y_shifted = [y + offset_y for y in sig_y]
                    
                    plt.scatter(sig_x_shifted, sig_y_shifted, color='black', marker='$*$', s=10, alpha=0.4, zorder=5, label='_nolegend_')

        ax.set_title(f'{x_metric} vs {y_metric} (nn={nn})')
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(f'{title_suffix}\n(* = significatività stat.)')
    
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels and l != "":
                handles.append(h)
                labels.append(l)

    fig.legend(handles, labels, bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize='small')
    plt.tight_layout()
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def plot_3d_combined(group_data_by_nn, x_metric, y_metric, z_metric, neighbor_values, all_labels=None, title_suffix="", output_file=None):
    from plotly.subplots import make_subplots
    if not any(group_data_by_nn.values()): return
    
    fig = make_subplots(
        rows=1, cols=len(neighbor_values),
        specs=[[{'type': 'scatter3d'} for _ in neighbor_values]],
        subplot_titles=[f"nn={nn}" for nn in neighbor_values]
    )

    if all_labels is None:
        all_labels = set()
        for nn, m_data in group_data_by_nn.items():
            all_labels.update(m_data.keys())
        all_labels = list(all_labels)
        
    base_labels = sorted(set(get_base_label(l) for l in all_labels))
    import plotly.colors as pcolors
    colors = pcolors.qualitative.Plotly
    if len(base_labels) > 10:
        colors = pcolors.qualitative.Alphabet
    markers = ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']

    show_legend = set()

    for i, nn in enumerate(neighbor_values):
        model_data = group_data_by_nn.get(nn, {})
        for label, data in model_data.items():
            base = get_base_label(label)
            idx = base_labels.index(base)
            c = colors[idx % len(colors)]
            m = markers[idx % len(markers)]
            
            x_sig_key, y_sig_key, z_sig_key = x_metric + "_sig", y_metric + "_sig", z_metric + "_sig"
            combined = sorted(zip(data[x_metric], data[y_metric], data[z_metric], data[x_sig_key], data[y_sig_key], data[z_sig_key]), key=lambda k: k[0])
            
            if combined:
                x_v, y_v, z_v, x_s, y_s, z_s = zip(*combined)
                
                show_l = label not in show_legend
                if show_l:
                    show_legend.add(label)

                fig.add_trace(go.Scatter3d(
                    x=x_v, y=y_v, z=z_v, mode='lines+markers', name=label,
                    marker=dict(size=4, color=c, symbol=m), line=dict(width=4, color=c),
                    text=[f"{label}<br>{x_metric}:{x:.3f}<br>{y_metric}:{y:.3f}<br>{z_metric}:{z:.3f}" for x, y, z in
                          zip(x_v, y_v, z_v)],
                    showlegend=show_l
                ), row=1, col=i+1)

                # Highlight significant points with a diamond in 3D
                sig_x = [x for x, y, z, xs, ys, zs in combined if xs or ys or zs]
                sig_y = [y for x, y, z, xs, ys, zs in combined if xs or ys or zs]
                sig_z = [z for x, y, z, xs, ys, zs in combined if xs or ys or zs]
                
                if sig_x:
                    x_range = max(max(x_v) - min(x_v), 0.001)
                    y_range = max(max(y_v) - min(y_v), 0.001)
                    z_range = max(max(z_v) - min(z_v), 0.001)
                    
                    sig_x_shifted = [x + x_range * 0.015 for x in sig_x]
                    sig_y_shifted = [y + y_range * 0.015 for y in sig_y]
                    sig_z_shifted = [z + z_range * 0.015 for z in sig_z]
                    
                    fig.add_trace(go.Scatter3d(
                        x=sig_x_shifted, y=sig_y_shifted, z=sig_z_shifted, mode='markers', name=f"{label} (Sig)",
                        marker=dict(size=6, color='black', symbol='diamond', opacity=0.8),
                        showlegend=False
                    ), row=1, col=i+1)

    fig.update_layout(
        title=f'3D: {x_metric} vs {y_metric} vs {z_metric} {title_suffix}<br>(* diamonds = significatività stat.)',
        margin=dict(l=0, r=0, b=0, t=50), legend=dict(x=0, y=1)
    )
    
    for i in range(1, len(neighbor_values) + 1):
        suffix = str(i) if i > 1 else ""
        if hasattr(fig.layout, f'scene{suffix}'):
            getattr(fig.layout, f'scene{suffix}').xaxis.title = x_metric
            getattr(fig.layout, f'scene{suffix}').yaxis.title = y_metric
            getattr(fig.layout, f'scene{suffix}').zaxis.title = z_metric
        
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if not output_file.endswith(".html"): output_file += ".html"
        fig.write_html(output_file)


def plot_3d(model_data, x_metric, y_metric, z_metric, neighbors, all_labels=None, title_suffix="", output_file=None):
    if not model_data: return
    fig = go.Figure()

    if all_labels is None:
        all_labels = list(model_data.keys())
        
    base_labels = sorted(set(get_base_label(l) for l in all_labels))
    import plotly.colors as pcolors
    colors = pcolors.qualitative.Plotly
    if len(base_labels) > 10:
        colors = pcolors.qualitative.Alphabet
    markers = ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']


    for label, data in model_data.items():
        base = get_base_label(label)
        idx = base_labels.index(base)
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        
        x_sig_key, y_sig_key, z_sig_key = x_metric + "_sig", y_metric + "_sig", z_metric + "_sig"
        combined = sorted(zip(data[x_metric], data[y_metric], data[z_metric], data[x_sig_key], data[y_sig_key], data[z_sig_key]), key=lambda k: k[0])
        
        if combined:
            x_v, y_v, z_v, x_s, y_s, z_s = zip(*combined)
            fig.add_trace(go.Scatter3d(
                x=x_v, y=y_v, z=z_v, mode='lines+markers', name=label,
                marker=dict(size=4, color=c, symbol=m), line=dict(width=4, color=c),
                text=[f"{label}<br>{x_metric}:{x:.3f}<br>{y_metric}:{y:.3f}<br>{z_metric}:{z:.3f}" for x, y, z in
                      zip(x_v, y_v, z_v)]
            ))
            
            # Highlight significant points with a star in 3D
            sig_x = [x for x, y, z, xs, ys, zs in combined if xs or ys or zs]
            sig_y = [y for x, y, z, xs, ys, zs in combined if xs or ys or zs]
            sig_z = [z for x, y, z, xs, ys, zs in combined if xs or ys or zs]
            
            if sig_x:
                x_range = max(max(x_v) - min(x_v), 0.001)
                y_range = max(max(y_v) - min(y_v), 0.001)
                z_range = max(max(z_v) - min(z_v), 0.001)
                
                sig_x_shifted = [x + x_range * 0.015 for x in sig_x]
                sig_y_shifted = [y + y_range * 0.015 for y in sig_y]
                sig_z_shifted = [z + z_range * 0.015 for z in sig_z]
                
                fig.add_trace(go.Scatter3d(
                    x=sig_x_shifted, y=sig_y_shifted, z=sig_z_shifted, mode='markers', name=f"{label} (Sig)",
                    marker=dict(size=6, color='black', symbol='diamond', opacity=0.8), # 'star' is not a valid 3D symbol in Plotly, using 'diamond'
                    showlegend=False
                ))

    fig.update_layout(
        title=f'3D: {x_metric} vs {y_metric} vs {z_metric} (nn={neighbors}) {title_suffix}<br>(* diamonds = significatività stat.)',
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

def run_automation(data_dir, metrics, split_sim, split_type, combined_nn=False):
    neighbor_values = [50, 100, 250]
    is_3d = (len(metrics) == 3)

    if is_3d:
        main_folder = f"{metrics[0]}_vs_{metrics[1]}_vs_{metrics[2]}"
    else:
        main_folder = f"{metrics[0]}_vs_{metrics[1]}"

    if combined_nn:
        print(f"Inizio generazione batch (combined) per: {main_folder}")
        all_data_by_nn = {}
        for nn in neighbor_values:
            model_data = load_data(data_dir, metrics, nn)
            all_data_by_nn[nn] = model_data

        groups_combined = {}
        if split_sim:
            for g_name, g_cond in [
                ("User_Cosine", lambda v: v['type'] == 'User' and v['sim'].lower() in ['cosine', 'angular']),
                ("User_Jaccard", lambda v: v['type'] == 'User' and v['sim'].lower() == 'jaccard'),
                ("Item_Cosine", lambda v: v['type'] == 'Item' and v['sim'].lower() in ['cosine', 'angular']),
                ("Item_Jaccard", lambda v: v['type'] == 'Item' and v['sim'].lower() == 'jaccard')
            ]:
                groups_combined[g_name] = {nn: {k: v for k, v in all_data_by_nn[nn].items() if g_cond(v)} for nn in neighbor_values if nn in all_data_by_nn}
        elif split_type:
            for g_name, g_cond in [
                ("User", lambda v: v['type'] == 'User'),
                ("Item", lambda v: v['type'] == 'Item')
            ]:
                groups_combined[g_name] = {nn: {k: v for k, v in all_data_by_nn[nn].items() if g_cond(v)} for nn in neighbor_values if nn in all_data_by_nn}
        else:
            groups_combined["All_Models"] = all_data_by_nn

        base_out_dir = os.path.join("plots", main_folder, "combined_nn")
        os.makedirs(base_out_dir, exist_ok=True)
        
        for g_name, g_data_by_nn in groups_combined.items():
            if not any(g_data_by_nn.values()): continue
            
            all_labels = set()
            for nn, md in g_data_by_nn.items():
                all_labels.update(md.keys())
            all_labels = list(all_labels)
            
            ext = ".html" if is_3d else ".png"
            out_path = os.path.join(base_out_dir, f"{g_name}{ext}")

            if is_3d:
                plot_3d_combined(g_data_by_nn, metrics[0], metrics[1], metrics[2], neighbor_values, all_labels=all_labels, title_suffix=f"({g_name})", output_file=out_path)
            else:
                plot_2d_combined(g_data_by_nn, metrics[0], metrics[1], neighbor_values, all_labels=all_labels, title_suffix=f"({g_name})", output_file=out_path)
        return

    print(f"Inizio generazione batch per: {main_folder}")

    for nn in neighbor_values:
        print(f"  > Processing nn={nn}")
        model_data = load_data(data_dir, metrics, nn)

        if not model_data:
            continue

        base_out_dir = os.path.join("plots", main_folder, f"nn_{nn}")
        all_labels = list(model_data.keys())

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
                plot_3d(g_data, metrics[0], metrics[1], metrics[2], nn, all_labels=all_labels, title_suffix=f"({g_name})",
                        output_file=out_path)
            else:
                plot_2d(g_data, metrics[0], metrics[1], nn, all_labels=all_labels, title_suffix=f"({g_name})", output_file=out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--metrics", nargs='+', required=True, help="2 o 3 metriche (X Y [Z])")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--split_sim", action="store_true")
    parser.add_argument("--combined_nn", action="store_true", help="Plot all neighbors in one figure with subplots")
    args = parser.parse_args()

    if len(args.metrics) not in [2, 3]:
        print("Errore: specifica 2 o 3 metriche.")
    else:
        run_automation(args.data_dir, args.metrics, args.split_sim, args.split, args.combined_nn)
        print("\nBatch completato. Controlla la cartella 'plots'.")