import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from pymoo.indicators.hv import HV
import os

class ParetoPlotter3D:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.df = None

    def load_data(self):
        """Reads all CSVs in data_folder and consolidates them."""
        all_files = list(self.data_folder.glob("*.csv"))
        df_list = []
        for filename in all_files:
            try:
                df = pd.read_csv(filename)
                df_list.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        if not df_list:
            raise ValueError(f"No CSV files found in {self.data_folder}")

        self.df = pd.concat(df_list, ignore_index=True)
        
        # Differentiate based on sampling strategy if 'strat' column exists
        if 'strat' in self.df.columns:
            def update_algo_name(row):
                algo = str(row['Algorithm'])
                strat = row.get('strat')
                # Check if strat is valid (not NaN and not empty)
                if pd.notna(strat) and str(strat).strip() != '':
                     return f"{algo}_{strat}"
                return algo
            
            self.df['Algorithm'] = self.df.apply(update_algo_name, axis=1)
        
        # Add Type column based on Algorithm name
        # Assuming User models start with 'User' and Item models with 'Item'
        def get_type(algo):
            if str(algo).startswith('User'):
                return 'User'
            elif str(algo).startswith('Item'):
                return 'Item'
            else:
                return 'Other'
        
        self.df['Type'] = self.df['Algorithm'].apply(get_type)

    def _get_pareto_frontier(self, group_df, metrics, directions):
        """
        Identifies the Pareto frontier for a given DataFrame and metrics.
        metrics: list of metric names
        directions: list of 'max' or 'min' corresponding to metrics
        """
        # Sort by first metric to make it easier, though not strictly necessary for the logic below
        # but good for plotting lines if we want them ordered
        ascending = (directions[0] == 'min')
        group_df = group_df.sort_values(by=metrics[0], ascending=ascending)
        
        pareto_front = []
        # Naive implementation O(N^2) - adequate for typically small number of points per model configuration
        # For larger datasets, a more efficient algorithm might be needed.
        # However, for just plotting a few points per model, this checks if a point is dominated
        
        points = group_df[metrics].values
        indices = group_df.index.tolist()
        
        is_pareto = [True] * len(points)
        
        for i in range(len(points)):
            for j in range(len(points)):
                if i == j:
                    continue
                # If point j dominates point i
                # Dominate means j is better or equal in all metrics AND strictly better in at least one
                dominates = True
                strictly_better = False
                
                for k in range(len(metrics)):
                    if directions[k] == 'max':
                        if points[j][k] < points[i][k]:
                            dominates = False
                            break
                        if points[j][k] > points[i][k]:
                            strictly_better = True
                    elif directions[k] == 'min':
                        if points[j][k] > points[i][k]:
                            dominates = False
                            break
                        if points[j][k] < points[i][k]:
                            strictly_better = True
                    else:
                        raise ValueError(f"Invalid direction '{directions[k]}'. Must be 'max' or 'min'.")
                
                if dominates and strictly_better:
                    is_pareto[i] = False
                    break
        
        return group_df.loc[[indices[i] for i in range(len(is_pareto)) if is_pareto[i]]]

    def plot_pareto_3d(self, metrics, directions=None, output_file=None):
        """
        Generates 3D plots for User and Item models.
        metrics: list of 3 metric names
        directions: list of 3 strings, 'max' or 'min'. Defaults to ['max', 'max', 'max'].
        """
        if len(metrics) != 3:
            raise ValueError("Exactly 3 metrics must be specified for 3D plotting.")

        if directions is None:
            directions = ['max', 'max', 'max']
        
        if len(directions) != 3:
            raise ValueError("Exactly 3 directions must be specified.")

        if self.df is None:
            self.load_data()

        types = ['User', 'Item']
        
        for model_type in types:
            type_df = self.df[self.df['Type'] == model_type]
            
            if type_df.empty:
                print(f"No data found for {model_type} models.")
                continue
                
            fig = go.Figure()
            
            # Group by Algorithm to plot separate traces
            for algo, algo_df in type_df.groupby('Algorithm'):
                # Calculate Pareto frontier for this algorithm
                # We want the pareto front OF THE ALGORITHM's configurations
                pareto_df = self._get_pareto_frontier(algo_df, metrics, directions)
                
                # Sort for better line connection if desired, usually by the first metric
                ascending = (directions[0] == 'min')
                pareto_df = pareto_df.sort_values(by=metrics[0], ascending=ascending)

                fig.add_trace(go.Scatter3d(
                    x=pareto_df[metrics[0]],
                    y=pareto_df[metrics[1]],
                    z=pareto_df[metrics[2]],
                    mode='lines+markers',
                    name=algo,
                    marker=dict(size=5),
                    text=[f"{algo}<br>{metrics[0]}: {x:.4f}<br>{metrics[1]}: {y:.4f}<br>{metrics[2]}: {z:.4f}" 
                          for x, y, z in zip(pareto_df[metrics[0]], pareto_df[metrics[1]], pareto_df[metrics[2]])]
                ))

            fig.update_layout(
                title=f'3D Pareto Frontier - {model_type} Models<br>Directions: {directions}',
                scene=dict(
                    xaxis_title=f"{metrics[0]} ({directions[0]})",
                    yaxis_title=f"{metrics[1]} ({directions[1]})",
                    zaxis_title=f"{metrics[2]} ({directions[2]})"
                ),
                margin=dict(l=0, r=0, b=0, t=50)
            )
            
            if output_file:
                # Append type to filename
                base, ext = os.path.splitext(output_file)
                filename = f"{base}_{model_type}{ext}"
                fig.write_html(filename)
                print(f"Saved {model_type} plot to {filename}")
            else:
                fig.show()

    import pandas as pd
    import numpy as np
    from pymoo.indicators.hv import HV

    def calculate_hypervolume(self, metrics, directions=None, output_file=None):
        """
        Calculates the true Hypervolume indicator for the Pareto frontier of each algorithm.
        Uses the 'pymoo' library to compute the union volume of the non-dominated points.
        """
        if self.df is None:
            self.load_data()

        if directions is None:
            directions = ['max', 'max', 'max']

        if len(metrics) != 3 or len(directions) != 3:
            raise ValueError("Metrics and directions must have length 3.")

        # 1. Calculate Reference Point (Worst Case + Offset)
        ref_point = []
        for i, metric in enumerate(metrics):
            valid_values = self.df[metric].dropna()
            if valid_values.empty:
                ref_point.append(0.0)
                continue

            if directions[i] == 'max':
                # Worst case for maximization is the minimum value.
                # We want the reference point to be slightly WORSE (smaller) than the minimum.
                # worst_val = valid_values.min()
                # if worst_val > 0:
                #     ref_point.append(worst_val * 0.99)
                # elif worst_val < 0:
                #     ref_point.append(worst_val * 1.01)
                # else:
                #     ref_point.append(-0.01)  # If exactly 0
                ref_point.append(-0.01)
            else:
                # Worst case for minimization is the maximum value.
                # We want the reference point to be slightly WORSE (larger) than the maximum.
                # worst_val = valid_values.max()
                # if worst_val > 0:
                #     ref_point.append(worst_val * 1.01)
                # elif worst_val < 0:
                #     ref_point.append(worst_val * 0.99)
                # else:
                #     ref_point.append(0.01)  # If exactly 0
                ref_point.append(1.01)


        print(f"Reference Point for Hypervolume (Original Scale): {dict(zip(metrics, ref_point))}")

        # Prepare reference point for Pymoo (which assumes strict minimization)
        # So 'max' metrics must be inverted (multiplied by -1)
        pymoo_ref = []
        for i in range(len(metrics)):
            if directions[i] == 'max':
                pymoo_ref.append(-ref_point[i])
            else:
                pymoo_ref.append(ref_point[i])

        ref_array = np.array(pymoo_ref)

        # Initialize the Hypervolume indicator with our reference point
        hv_indicator = HV(ref_point=ref_array)

        results = []
        types = ['User', 'Item']  # Adjust if your 'Type' column contains other categories

        for model_type in types:
            type_df = self.df[self.df['Type'] == model_type]
            if type_df.empty:
                continue

            for algo, algo_df in type_df.groupby('Algorithm'):
                # Get the Pareto front points for this specific algorithm
                pareto_df = self._get_pareto_frontier(algo_df, metrics, directions)

                if pareto_df.empty:
                    continue

                # 2. Extract points and prepare them for Pymoo
                front_points = []
                for _, row in pareto_df.iterrows():
                    point = []
                    for i, metric in enumerate(metrics):
                        val = row[metric]
                        if directions[i] == 'max':
                            point.append(-val)  # Invert for minimization
                        else:
                            point.append(val)
                    front_points.append(point)

                front_array = np.array(front_points)

                # 3. Calculate the Total Hypervolume of the front
                try:
                    # ind(front_array) calculates the volume of the union of all dominated hypercubes
                    total_hv = hv_indicator(front_array)
                except Exception as e:
                    print(f"Could not compute HV for {algo}: {e}")
                    total_hv = 0.0

                # Store exactly ONE result per algorithm
                results.append({
                    'Type': model_type,
                    'Algorithm': algo,
                    'Hypervolume': total_hv,
                    'Points_in_Frontier': len(pareto_df),
                    'Total_Points': len(algo_df)  # <--- NEW COLUMN ADDED HERE
                })

        results_df = pd.DataFrame(results)

        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Saved Hypervolume results to {output_file}")

        return results_df