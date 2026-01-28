from pathlib import Path
from result_analyzer.multi_object_analisys.pareto_front import mobj_pipeline
from result_analyzer.dataset_cleaner.utils_dataset import create_integrated_dataset

# --------- main -------------------
config_name = "ml_100k_kiwiat"

target_folder = (
    f"/Users/pierluromani/Desktop/progetto_elliot/elliot_nat/elliot/results/{config_name}/performance"
)
df_final = create_integrated_dataset(target_folder)
csv_path = Path(
    f"/Users/pierluromani/Desktop/progetto_elliot/elliot_nat/elliot/results/{config_name}/performance/{config_name}_dataset.csv"
)
accuracy_metrics = [
    "nDCGRendle2020",
    "Recall",
    "HR",
    "Precision",
    # "MAP",
    "MRR",
]
beyond_accuracy = [
    "ItemCoverage",
    # "UserCoverage",
    # "ACLT",
    "Gini",
    "SEntropy",
    "EFD",
    "EPC",
    "PopREO",
    # "PopRSP",
]
mobj_pipeline(csv_path, accuracy_metrics, beyond_accuracy, save=True)