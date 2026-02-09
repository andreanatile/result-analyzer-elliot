from pathlib import Path
from result_analyzer.multi_object_analisys.pareto_front import mobj_pipeline
from result_analyzer.dataset_cleaner.utils_dataset import create_integrated_dataset

# --------- main -------------------
folder_name = "movielens_100k_item"

target_folder = (
    f"/Users/pierluromani/Desktop/progetto_elliot/elliot-ANN2.0/results/{folder_name}/performance"
)
df_final = create_integrated_dataset(target_folder, f'{folder_name}.csv')
csv_path = Path(
    f"/Users/pierluromani/Desktop/progetto_elliot/elliot-ANN2.0/results/{folder_name}/performance/{folder_name}.csv"
)
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
mobj_pipeline(csv_path, accuracy_metrics, beyond_accuracy, save=True)