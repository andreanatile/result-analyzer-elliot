import os
import re
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go

try:
    from pymoo.indicators.hv import HV
except ImportError:
    HV = None


class ParetoPlotter3D:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.df = None
        # Global styling dictionaries
        self.color_map = {}
        self.marker_2d_map = {}
        self.marker_3d_map = {}

    def load_data(self):
        """Reads all CSVs in data_folder and consolidates them."""
        all_files = list(self.data_folder.glob("*.csv"))
        df_list = []
        for filename in all_files:
            try:
                df = pd.read_csv(filename)

                # Extract threshold from filename
                match = re.search(r'(?:t|threshold)_?(\d+(?:_|\.)\d+)', filename.name, re.IGNORECASE)
                if match:
                    thres_str = match.group(1).replace('_', '.')
                    df['threshold'] = float(thres_str)
                else:
                    if 'threshold' not in df.columns:
                        df['threshold'] = None

                df_list.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        if not df_list:
            raise ValueError(f"No CSV files found in {self.data_folder}")

        self.df = pd.concat(df_list, ignore_index=True)

        # Handle the 'sim' column and convert 'angular' to 'cosine'
        if 'sim' in self.df.columns:
            self.df['sim'] = self.df['sim'].apply(
                lambda x: 'cosine' if str(x).lower() == 'angular' else str(x) if pd.notna(x) else np.nan
            )

        # Differentiate based on sampling strategy
        if 'strat' in self.df.columns:
            def update_algo_name(row):
                algo = str(row['Algorithm'])
                strat = row.get('strat')
                if pd.notna(strat) and str(strat).strip() != '':
                    return f"{algo}_{strat}"
                return algo

            self.df['Algorithm'] = self.df.apply(update_algo_name, axis=1)

        # Ensure 'Type' column exists
        if 'Type' not in self.df.columns:
            self.df['Type'] = self.df['Algorithm'].apply(
                lambda x: 'User' if 'user' in str(x).lower() else 'Item'
            )

        # --- Display-name overrides ---
        # FairANN_no_sampling is renamed based on its similarity metric
        if 'sim' in self.df.columns:
            def _rename_fairann(row):
                algo = str(row['Algorithm'])
                sim  = str(row.get('sim', '')).lower()
                # Extract Item/User prefix
                if algo.startswith('Item'):
                    prefix, base = 'Item', algo[4:]
                elif algo.startswith('User'):
                    prefix, base = 'User', algo[4:]
                else:
                    prefix, base = '', algo
                if 'fairann_no_sampling' in base.lower():
                    if sim == 'jaccard':
                        return f"{prefix}Minhashing"
                    if sim in ('cosine', 'angular'):
                        return f"{prefix}LSHRandomProjection"
                return algo

            self.df['Algorithm'] = self.df.apply(_rename_fairann, axis=1)

    def get_base_label(self, label):
        """Extracts the base model name for consistent styling."""
        base = str(label)
        if base.startswith('User'): base = base[4:]
        if base.startswith('Item'): base = base[4:]
        base = base.replace(' - cosine', '').replace(' - jaccard', '').replace(' - angular', '')
        # Optional: clean up trailing underscores if they appear
        base = base.strip('_')
        return base

    def _prepare_global_styles(self):
        """Generates consistent colors and markers based on the base algorithm name."""
        all_labels = self.df['Algorithm'].dropna().unique()
        base_labels = sorted(set(self.get_base_label(l) for l in all_labels))

        # Select colormap based on number of unique base algorithms
        if len(base_labels) > 10:
            colors_rgb = plt.cm.tab20.colors
        else:
            colors_rgb = plt.cm.tab10.colors

        # Convert RGB to HEX to ensure perfect consistency between Matplotlib and Plotly
        hex_colors = [mcolors.to_hex(c) for c in colors_rgb]

        # Define markers for 2D (Matplotlib)
        markers_2d = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', 'd']

        # Define equivalent markers for 3D (Plotly Scatter3d has a limited set)
        markers_3d = ['circle', 'square', 'diamond', 'cross', 'x', 'circle-open', 'square-open', 'diamond-open']

        for i, base in enumerate(base_labels):
            self.color_map[base] = hex_colors[i % len(hex_colors)]
            self.marker_2d_map[base] = markers_2d[i % len(markers_2d)]
            self.marker_3d_map[base] = markers_3d[i % len(markers_3d)]

    def _get_pareto_frontier(self, df, metrics, directions):
        """Calculates the Pareto frontier for a given subset of data."""
        points = df[metrics].values
        converted_points = np.copy(points)

        for i, d in enumerate(directions):
            if d == 'min':
                converted_points[:, i] = -converted_points[:, i]

        is_pareto = np.ones(converted_points.shape[0], dtype=bool)
        for i, c in enumerate(converted_points):
            if is_pareto[i]:
                is_pareto[is_pareto] = np.any(converted_points[is_pareto] > c, axis=1) | np.all(
                    converted_points[is_pareto] == c, axis=1)
                is_pareto[i] = True

        return df.iloc[is_pareto].copy()

    def plot_pareto(self, metrics, directions, split_sim=False, split_threshold=False, log_metrics=None,
                    output_file="pareto_plot"):
        """Main dispatcher that handles slicing the data and triggering 2D or 3D plots."""
        if self.df is None:
            self.load_data()

        # Build the global styling maps before generating any plots
        self._prepare_global_styles()

        if log_metrics is None:
            log_metrics = []

        threshold_col = 'threshold' if 'threshold' in self.df.columns else None

        sims = self.df['sim'].dropna().unique() if (split_sim and 'sim' in self.df.columns) else [None]
        thresholds = self.df[threshold_col].dropna().unique() if (split_threshold and threshold_col) else [None]

        model_types = self.df['Type'].dropna().unique()

        out_dir = os.path.dirname(output_file)
        if not out_dir:
            out_dir = "."

        metrics_subfolder = "_vs_".join(metrics) + '_pareto'
        base_plots_dir = os.path.join(out_dir, "plots", metrics_subfolder)

        base = os.path.basename(output_file)
        base_name, ext = os.path.splitext(base)

        for model_type in model_types:
            df_type = self.df[self.df['Type'] == model_type]
            if df_type.empty: continue

            for sim in sims:
                for thres in thresholds:
                    df_filtered = df_type.copy()

                    # Format threshold with 2 decimal places so 0.5 -> '0_50', 0.25 -> '0_25'
                    thres_str = f"{thres:.2f}".replace('.', '_') if thres is not None else None

                    current_plots_dir = base_plots_dir
                    if split_threshold and thres_str is not None:
                        current_plots_dir = os.path.join(base_plots_dir, f"threshold_{thres_str}")

                    name_parts = [model_type]

                    if sim is not None:
                        df_filtered = df_filtered[df_filtered['sim'] == sim]
                        name_parts.append(str(sim))

                    if thres is not None:
                        df_filtered = df_filtered[df_filtered[threshold_col] == thres]
                        name_parts.append(f"threshold_{thres_str}")

                    if df_filtered.empty:
                        continue

                    prefix = "_".join(name_parts)
                    title_suffix = prefix.replace("_", " ").title()

                    current_ext = ext
                    if len(metrics) == 2:
                        current_ext = ".png" if current_ext in ["", ".html"] else current_ext
                    else:
                        current_ext = ".html" if current_ext in ["", ".png"] else current_ext

                    out_name = os.path.join(current_plots_dir, f"{prefix}_{base_name}{current_ext}")

                    if len(metrics) == 2:
                        self._plot_2d(df_filtered, metrics, directions, title_suffix=title_suffix,
                                      log_metrics=log_metrics, output_file=out_name)
                    elif len(metrics) == 3:
                        self._plot_3d(df_filtered, metrics, directions, title_suffix=title_suffix,
                                      log_metrics=log_metrics, output_file=out_name)

    def _plot_2d(self, df_subset, metrics, directions, title_suffix="", log_metrics=None, output_file=None):
        """Generates a 2D Scatterplot using Matplotlib, drawing the Pareto line."""
        plt.figure(figsize=(10, 7))
        algorithms = df_subset['Algorithm'].unique()

        for algo in algorithms:
            algo_df = df_subset[df_subset['Algorithm'] == algo]
            if algo_df.empty: continue

            # Use consistent global styling
            base_algo = self.get_base_label(algo)
            c = self.color_map.get(base_algo, '#333333')
            m = self.marker_2d_map.get(base_algo, 'o')

            plt.scatter(algo_df[metrics[0]], algo_df[metrics[1]], color=c, marker=m, alpha=0.3, s=40)

            pareto_df = self._get_pareto_frontier(algo_df, metrics, directions)
            if not pareto_df.empty:
                pareto_df = pareto_df.sort_values(by=metrics[0])
                plt.plot(pareto_df[metrics[0]], pareto_df[metrics[1]], color=c, marker=m,
                         label=f"{algo}", linewidth=1.5)

        plt.xlabel(metrics[0])
        plt.ylabel(metrics[1])

        if log_metrics:
            if metrics[0] in log_metrics:
                plt.xscale('log')
            if metrics[1] in log_metrics:
                plt.yscale('log')

        plt.title(f"2D Pareto Frontier: {metrics[0]} vs {metrics[1]}\n({title_suffix})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Algorithms")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            plt.savefig(output_file, dpi=300)
            print(f"-> Saved 2D plot: {output_file}")

        plt.close()

    def _plot_3d(self, df_subset, metrics, directions, title_suffix="", log_metrics=None, output_file=None):
        """Generates a 3D Scatterplot using Plotly."""
        fig = go.Figure()
        algorithms = df_subset['Algorithm'].unique()

        for algo in algorithms:
            algo_df = df_subset[df_subset['Algorithm'] == algo]
            if algo_df.empty: continue

            # Use consistent global styling
            base_algo = self.get_base_label(algo)
            c = self.color_map.get(base_algo, '#333333')
            m = self.marker_3d_map.get(base_algo, 'circle')

            fig.add_trace(go.Scatter3d(
                x=algo_df[metrics[0]], y=algo_df[metrics[1]], z=algo_df[metrics[2]],
                mode='markers',
                marker=dict(size=3, color=c, symbol=m, opacity=0.3),
                name=f"{algo} (All)",
                legendgroup=algo,
                showlegend=False
            ))

            pareto_df = self._get_pareto_frontier(algo_df, metrics, directions)
            if not pareto_df.empty:
                pareto_df = pareto_df.sort_values(by=metrics[0])
                fig.add_trace(go.Scatter3d(
                    x=pareto_df[metrics[0]], y=pareto_df[metrics[1]], z=pareto_df[metrics[2]],
                    mode='lines+markers',
                    marker=dict(size=4, color=c, symbol=m),
                    line=dict(color=c, width=4),
                    name=f"{algo}",
                    legendgroup=algo,
                    text=[
                        f"{algo}<br>{metrics[0]}:{row[metrics[0]]:.3f}<br>{metrics[1]}:{row[metrics[1]]:.3f}<br>{metrics[2]}:{row[metrics[2]]:.3f}"
                        for _, row in pareto_df.iterrows()]
                ))

        scene_dict = dict(
            xaxis_title=metrics[0],
            yaxis_title=metrics[1],
            zaxis_title=metrics[2]
        )

        if log_metrics:
            if metrics[0] in log_metrics:
                scene_dict['xaxis_type'] = 'log'
            if metrics[1] in log_metrics:
                scene_dict['yaxis_type'] = 'log'
            if metrics[2] in log_metrics:
                scene_dict['zaxis_type'] = 'log'

        fig.update_layout(
            title=f"3D Pareto Frontier<br>({title_suffix})",
            scene=scene_dict,
            margin=dict(l=0, r=0, b=0, t=60),
            legend=dict(x=1.02, y=1)
        )

        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"-> Saved 3D plot: {output_file}")

    def calculate_hypervolume(self, metrics, directions=None, absolute_ref_point=None, output_file=None):
        """Calculates exact Hypervolume values using PyMoo."""
        if HV is None:
            print("Error: pymoo library is not installed. Run `pip install pymoo`.")
            return None

        if self.df is None:
            self.load_data()

        pymoo_ref = []
        for i in range(len(metrics)):
            if directions[i] == 'max':
                pymoo_ref.append(-absolute_ref_point[i])
            else:
                pymoo_ref.append(absolute_ref_point[i])

        ref_array = np.array(pymoo_ref)
        hv_indicator = HV(ref_point=ref_array)

        results = []
        groupby_cols = ['Algorithm', 'sim'] if 'sim' in self.df.columns else ['Algorithm']

        for group_keys, algo_df in self.df.groupby(groupby_cols, dropna=False):
            algo = group_keys[0] if isinstance(group_keys, tuple) else group_keys
            sim = group_keys[1] if isinstance(group_keys, tuple) else None
            display_name = f"{algo} ({sim})" if pd.notna(sim) else algo

            pareto_df = self._get_pareto_frontier(algo_df, metrics, directions)
            if pareto_df.empty:
                continue

            front_points = []
            for _, row in pareto_df.iterrows():
                point = []
                for i, metric in enumerate(metrics):
                    val = row[metric]
                    point.append(-val if directions[i] == 'max' else val)
                front_points.append(point)

            try:
                total_hv = hv_indicator(np.array(front_points))
            except Exception:
                total_hv = 0.0

            model_type = algo_df['Type'].iloc[0] if 'Type' in algo_df.columns else (
                'User' if 'user' in algo.lower() else 'Item')

            results.append({
                'Type': model_type,
                'Algorithm': algo,
                'Similarity': sim,
                'Algorithm_Variant': display_name,
                'Hypervolume': total_hv,
                'Points_in_Frontier': len(pareto_df),
                'Total_Points': len(algo_df)
            })

        results_df = pd.DataFrame(results).sort_values(by=['Type', 'Hypervolume'], ascending=[True, False])
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
            results_df.to_csv(output_file, index=False)
            print(f"-> Saved Hypervolume results to {output_file}")

        return results_df