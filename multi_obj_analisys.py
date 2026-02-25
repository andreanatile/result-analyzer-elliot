from pathlib import Path
from result_analyzer.multi_object_analisys.pareto_front import mobj_pipeline
from result_analyzer.dataset_cleaner.utils_dataset import create_integrated_dataset
from result_analyzer.multi_object_analisys.pareto_3d import ParetoPlotter3D

# --------- main -------------------
# Existing code commented out as paths are invalid in this environment
# folder_name = "movielens_100k_item"
# 
# target_folder = (
#     f"/Users/pierluromani/Desktop/progetto_elliot/elliot-ANN2.0/results/{folder_name}/performance"
# )
# # df_final = create_integrated_dataset(target_folder, f'{folder_name}.csv')
# csv_path = Path(
#     f"/Users/pierluromani/Desktop/progetto_elliot/elliot-ANN2.0/results/{folder_name}/performance/{folder_name}.csv"
# )
accuracy_metrics = [
    "nDCGRendle2020",
    # "Recall",
    # "HR",
    # "Precision",
    # # "MAP",
    # "MRR",
]
beyond_accuracy = [
    # "ItemCoverage",
    # "UserCoverage",
    # "ACLT",
    "Gini",
    # "SEntropy",
    # "EFD",
    "EPC",
    # "PopREO",
    # "PopRSP",
]
# mobj_pipeline(csv_path, accuracy_metrics, beyond_accuracy, save=True)

# ----------------- 3D Pareto Plotting Configuration -----------------
# You can modify these values to run the script directly with your preferred settings
DEFAULT_DATA_FOLDER = "./data"
DEFAULT_METRICS = ["nDCGRendle2020", "Recall", "Gini"]
DEFAULT_DIRECTIONS = ["max", "max", "min"] # 'max' or 'min'
DEFAULT_OUTPUT_FILE = "pareto_3d.html"

# ----------------- Main Execution -----------------
import argparse

if __name__ == "__main__":

    n_metrics = 2
    parser = argparse.ArgumentParser(description="3D Pareto Frontier Plotting")

    parser.add_argument("--data_folder", type=str, default=DEFAULT_DATA_FOLDER, 
                        help=f"Path to the data folder (default: {DEFAULT_DATA_FOLDER})")
    
    parser.add_argument("--metrics", nargs=n_metrics, default=DEFAULT_METRICS,
                        help=f"List of 3 metrics to plot (default: {DEFAULT_METRICS})")
    
    parser.add_argument("--directions", nargs=n_metrics, choices=['max', 'min'], default=DEFAULT_DIRECTIONS,
                        help=f"Optimization direction for each metric (default: {DEFAULT_DIRECTIONS})")
    
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE, 
                        help=f"Output HTML file name (default: {DEFAULT_OUTPUT_FILE})")

    parser.add_argument("--compute_hypervolume", action="store_true", 
                        help="Compute and save Hypervolume metrics for the Pareto frontier")

    args = parser.parse_args()

    print(f"Starting 3D Pareto Plotting...")
    print(f"Data Folder: {args.data_folder}")
    print(f"Metrics: {args.metrics}")
    print(f"Directions: {args.directions}")
    
    try:
        plotter = ParetoPlotter3D(args.data_folder)
        plotter.load_data()
        
        # Differentiate KNNFairness models by their 'preposp' method
        if 'preposp' in plotter.df.columns:
            mask = plotter.df['Algorithm'].str.contains('KNNfairness', case=False, na=False)
            has_preposp = plotter.df['preposp'].notna() & (plotter.df['preposp'] != '')
            plotter.df.loc[mask & has_preposp, 'Algorithm'] = plotter.df.loc[mask & has_preposp].apply(
                lambda row: f"{row['Algorithm']}_{row['preposp']}", axis=1
            )

        # plotter.plot_pareto_3d(args.metrics, directions=args.directions, output_file=args.output_file)
        
        if args.compute_hypervolume:
            print("Calculating Hypervolume...")
            hv_output_file = "efficency.csv"
            plotter.calculate_hypervolume(args.metrics, directions=args.directions, output_file=hv_output_file)

        print("Finished 3D Pareto Plotting.")
    except Exception as e:
        print(f"An error occurred: {e}")