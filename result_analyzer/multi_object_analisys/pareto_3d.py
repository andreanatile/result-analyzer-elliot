import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
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
