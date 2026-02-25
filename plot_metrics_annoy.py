import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_data(data_dir, metric):
    """
    Loads data from CSV files in the directory.
    Returns a single concatenated DataFrame.
    """
    all_dfs = []
    
    for filename in os.listdir(data_dir):
        if filename.startswith("target_t_") and filename.endswith("_annoy.csv"):
            try:
                t_str = filename.replace("target_t_", "").replace("_annoy.csv", "")
                t = float(t_str.replace("_", "."))
            except ValueError:
                print(f"Skipping file with unexpected name format: {filename}")
                continue
            
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                
                if metric not in df.columns or 'nn' not in df.columns or 'k' not in df.columns:
                    print(f"Skipping {filename}: Required columns missing.")
                    continue
                
                df['t'] = t
                # Determine type (User vs Item)
                df['model_type'] = df['Algorithm'].apply(
                    lambda x: 'User' if str(x).startswith('User') else ('Item' if str(x).startswith('Item') else 'Unknown')
                )
                
                all_dfs.append(df)

            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

def plot_data(df_plot, metric, m_type, nn, output_file=None):
    if df_plot.empty:
        return
        
    plt.figure(figsize=(12, 8))
    
    # Get unique search_k values
    search_ks = sorted(df_plot['k'].unique())
    
    # Generate colors and markers
    colors = plt.cm.tab10.colors
    if len(search_ks) > 10:
        colors = plt.cm.tab20.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'P', 'd']
    
    for idx, sk in enumerate(search_ks):
        subset = df_plot[df_plot['k'] == sk].sort_values(by='t')
        
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        
        label = f"search_k = {sk}"
        
        plt.scatter(subset['t'], subset[metric], label=label, marker=m, color=c, alpha=0.7, s=50)
        plt.plot(subset['t'], subset[metric], color=c, alpha=0.3)

    plt.title(f'{metric} vs t Threshold - {m_type} Models (Neighbors={nn})')
    plt.xlabel('t Threshold')
    plt.ylabel(metric)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    plt.close()

def plot_annoy_metrics(data_dir, metric, output_dir=None):
    df = load_data(data_dir, metric)
    if df.empty:
        print("No valid data found.")
        return
        
    if output_dir is None:
        output_dir = os.path.join("plots", "annoy", metric)
        
    # Split by user/item and nn
    for m_type in ['User', 'Item']:
        df_type = df[df['model_type'] == m_type]
        if df_type.empty:
            continue
            
        for nn in sorted(df_type['nn'].unique()):
            df_plot = df_type[df_type['nn'] == nn]
            
            output_file = os.path.join(output_dir, f"{metric}_{m_type}_nn{nn}.png")
            print(f"Plotting {m_type} models with nn={nn}...")
            plot_data(df_plot, metric, m_type, nn, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metric vs t threshold for ANNOY models.")
    parser.add_argument("--data_dir", type=str, default="data_annoy", help="Directory containing data files.")
    parser.add_argument("--metric", type=str, required=True, help="Metric to plot (e.g., Recall, nDCGRendle2020).")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' not found.")
    else:
        plot_annoy_metrics(args.data_dir, args.metric, args.output_dir)
