import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def load_data(data_dir, metric, neighbors):
    """
    Loads data from CSV files in the directory, filtering by neighbors.
    
    Returns:
        dict: Key: Label (str), Value: {'t': [], 'metric': [], 'type': 'User'|'Item'|'Unknown'}
    """
    model_data = {}
    
    # Iterate over files in the directory
    for filename in os.listdir(data_dir):
        if filename.startswith("target_t_") and filename.endswith(".csv") and "_minh" not in filename:
            # Extract t from filename
            try:
                t_str = filename.replace("target_t_", "").replace(".csv", "")
                t = float(t_str.replace("_", "."))
            except ValueError:
                print(f"Skipping file with unexpected name format: {filename}")
                continue
            
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                
                # Check if required columns exist
                if metric not in df.columns or 'nn' not in df.columns:
                    print(f"Skipping {filename}: 'nn' or '{metric}' column missing.")
                    continue
                
                # Filter by neighbors
                filtered_df = df[df['nn'] == neighbors].copy()
                
                if not filtered_df.empty:
                    for index, row in filtered_df.iterrows():
                        # Construct label: Algorithm - sim [- strat]
                        algo = row.get('Algorithm', 'Unknown')
                        sim = row.get('sim', 'Unknown')
                        strat = str(row.get('strat', '')).strip() if pd.notna(row.get('strat')) else ''
                        
                        # Custom naming for FairANN families when no strategy is present
                        algo_str = str(algo)
                        sim_str = str(sim).lower()
                        
                        if 'FairANN' in algo_str and strat == '':
                            prefix = 'User' if algo_str.startswith('User') else 'Item'
                            if sim_str == 'jaccard':
                                label = f"{prefix}Minhashing"
                            elif sim_str == 'cosine':
                                label = f"{prefix}LSHRandomProjection"
                            else:
                                # Fallback if sim is neither (unlikely based on request)
                                label = f"{algo} - {sim}"
                        else:
                            # Default naming
                            label_parts = [str(algo), str(sim)]
                            if strat:
                                label_parts.append(strat)
                            label = " - ".join(label_parts)
                        
                        # Determine type (User vs Item)
                        model_type = 'Unknown'
                        if str(algo).startswith('User'):
                            model_type = 'User'
                        elif str(algo).startswith('Item'):
                            model_type = 'Item'

                        if label not in model_data:
                            model_data[label] = {
                                't': [], 
                                'metric': [], 
                                'type': model_type,
                                'sim': str(sim),
                                'algo': str(algo)
                            }
                        
                        model_data[label]['t'].append(t)
                        model_data[label]['metric'].append(row[metric])

            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    return model_data

def plot_data(model_data, metric, neighbors, title_suffix="", output_file=None):
    """
    Plots the data provided in model_data.
    """
    if not model_data:
        print(f"No data to plot for {title_suffix}.")
        return

    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, (label, data) in enumerate(model_data.items()):
        sorted_indices = sorted(range(len(data['t'])), key=lambda k: data['t'][k])
        t_sorted = [data['t'][k] for k in sorted_indices]
        metric_sorted = [data['metric'][k] for k in sorted_indices]
        
        marker = markers[i % len(markers)]
        plt.scatter(t_sorted, metric_sorted, label=label, marker=marker, alpha=0.7, s=50)
        plt.plot(t_sorted, metric_sorted, alpha=0.3)

    plt.title(f'{metric} vs t Threshold (Neighbors={neighbors}) {title_suffix}')
    plt.xlabel('t Threshold')
    plt.ylabel(metric)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
            
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    plt.close()

def plot_metric_by_threshold(data_dir, metric, neighbors, output_file=None):
    model_data = load_data(data_dir, metric, neighbors)
    plot_data(model_data, metric, neighbors, output_file=output_file)

def plot_metric_split_by_user_item(data_dir, metric, neighbors, output_prefix=None):
    model_data = load_data(data_dir, metric, neighbors)
    
    user_data = {k: v for k, v in model_data.items() if v['type'] == 'User'}
    item_data = {k: v for k, v in model_data.items() if v['type'] == 'Item'}
    
    if output_prefix:
        output_user = f"{output_prefix}_User.png"
        output_item = f"{output_prefix}_Item.png"
    else:
        output_user = None
        output_item = None
        
    print("Plotting User Models...")
    plot_data(user_data, metric, neighbors, title_suffix="(User Models)", output_file=output_user)
    
    print("Plotting Item Models...")
    plot_data(item_data, metric, neighbors, title_suffix="(Item Models)", output_file=output_item)

def plot_metric_split_by_sim(data_dir, metric, neighbors, output_prefix=None):
    model_data = load_data(data_dir, metric, neighbors)
    
    # helper to check if algo is Annoy
    def is_annoy(algo_name):
        return 'ANNOY' in algo_name.upper()

    # Split into 4 groups
    user_cosine = {}
    user_jaccard = {}
    item_cosine = {}
    item_jaccard = {}
    
    for label, data in model_data.items():
        m_type = data['type']
        sim = data['sim'].lower()
        algo = data['algo']
        
        # Annoy goes to both Cosine and Jaccard lists
        # Others go to their respective list based on sim
        
        if m_type == 'User':
            if sim == 'cosine':
                user_cosine[label] = data
            elif sim == 'jaccard':
                user_jaccard[label] = data
            elif is_annoy(algo) or sim == 'angular':
                user_cosine[label] = data
                user_jaccard[label] = data
                
        elif m_type == 'Item':
            if sim == 'cosine':
                item_cosine[label] = data
            elif sim == 'jaccard':
                item_jaccard[label] = data
            elif is_annoy(algo) or sim == 'angular':
                item_cosine[label] = data
                item_jaccard[label] = data

    if output_prefix:
        p_user_cos = f"{output_prefix}_User_Cosine.png"
        p_user_jac = f"{output_prefix}_User_Jaccard.png"
        p_item_cos = f"{output_prefix}_Item_Cosine.png"
        p_item_jac = f"{output_prefix}_Item_Jaccard.png"
    else:
        p_user_cos = p_user_jac = p_item_cos = p_item_jac = None

    print("Plotting User Models (Cosine + Annoy)...")
    plot_data(user_cosine, metric, neighbors, title_suffix="(User - Cosine)", output_file=p_user_cos)
    
    print("Plotting User Models (Jaccard + Annoy)...")
    plot_data(user_jaccard, metric, neighbors, title_suffix="(User - Jaccard)", output_file=p_user_jac)
    
    print("Plotting Item Models (Cosine + Annoy)...")
    plot_data(item_cosine, metric, neighbors, title_suffix="(Item - Cosine)", output_file=p_item_cos)
    
    print("Plotting Item Models (Jaccard + Annoy)...")
    plot_data(item_jaccard, metric, neighbors, title_suffix="(Item - Jaccard)", output_file=p_item_jac)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot metric vs t threshold.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data files.")
    parser.add_argument("--metric", type=str, required=True, help="Metric to plot (e.g., Recall, nDCGRendle2020).")
    parser.add_argument("--neighbors", type=int, required=True, help="Number of neighbors (k/nn).")
    parser.add_argument("--output", type=str, default=None, help="Output file (or prefix for split). Default: plots/{metric}_{neighbors}/plot.png")
    parser.add_argument("--split", action="store_true", help="Split into User and Item plots.")
    parser.add_argument("--split_sim", action="store_true", help="Split into User/Item AND Cosine/Jaccard plots.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory '{args.data_dir}' not found.")
    else:
        # Handle default output if not specified
        if args.output is None:
            output_folder = f"{args.metric}_{args.neighbors}"
            output_dir = os.path.join("plots", output_folder)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created default output directory: {output_dir}")
            args.output = os.path.join(output_dir, "plot.png")

        # If output includes extension, strip it for prefix usage in split modes
        output_prefix = args.output
        if output_prefix.endswith(".png"):
            output_prefix = output_prefix[:-4]
            
        if args.split_sim:
            plot_metric_split_by_sim(args.data_dir, args.metric, args.neighbors, output_prefix)
        elif args.split:
            plot_metric_split_by_user_item(args.data_dir, args.metric, args.neighbors, output_prefix)
        else:
            plot_metric_by_threshold(args.data_dir, args.metric, args.neighbors, args.output)
