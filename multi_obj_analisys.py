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
    parser = argparse.ArgumentParser(description="Multi-Objective Analysis and Pareto Plotting")

    parser.add_argument("--data_folder", type=str, required=True,
                        help="Folder containing CSV files")
    parser.add_argument("--metrics", nargs='+', required=True,
                        help="List of 2 or 3 metrics to plot (e.g., nDCG Gini)")
    parser.add_argument("--directions", nargs='+', required=True,
                        help="Directions ('max' or 'min') for each metric")
    parser.add_argument("--output_file", type=str, default="pareto_plot",
                        help="Base name for the output file(s)")

    # --- NEW FLAG for Log Scale ---
    parser.add_argument("--log_metrics", nargs='+', default=[],
                        help="List of metrics to plot on a logarithmic scale (e.g., --log_metrics Gini)")

    parser.add_argument("--split_threshold", action="store_true",
                        help="Generate separate plots for each target threshold value")
    parser.add_argument("--split_sim", action="store_true",
                        help="Generate separate plots by similarity metric (cosine/jaccard)")
    parser.add_argument("--compute_hypervolume", action="store_true",
                        help="Compute and save Hypervolume metrics for the Pareto frontier")

    args = parser.parse_args()

    print(f"Starting Pareto Plotting...")
    print(f"Data Folder: {args.data_folder}")
    print(f"Metrics: {args.metrics}")
    print(f"Directions: {args.directions}")
    if args.log_metrics:
        print(f"Logarithmic Scale applied to: {args.log_metrics}")

    if len(args.metrics) not in [2, 3]:
        raise ValueError("Error: You must provide exactly 2 or 3 metrics for plotting.")
    if len(args.metrics) != len(args.directions):
        raise ValueError("Error: The number of metrics and directions must be identical.")

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

        # Plot the frontiers with log scaling capability
        plotter.plot_pareto(
            metrics=args.metrics,
            directions=args.directions,
            split_sim=args.split_sim,
            split_threshold=args.split_threshold,
            log_metrics=args.log_metrics,
            output_file=args.output_file
        )
        
        if args.compute_hypervolume:
            print("Calculating Hypervolume...")
            hv_output_file = "efficency.csv"
            plotter.calculate_hypervolume(args.metrics, directions=args.directions, output_file=hv_output_file)

        print("Finished 3D Pareto Plotting.")
    except Exception as e:
        print(f"An error occurred: {e}")