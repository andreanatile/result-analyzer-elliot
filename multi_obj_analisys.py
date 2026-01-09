from pathlib import Path
from result_analyzer.multi_object_analisys.pareto_front import mobj_pipeline
from result_analyzer.dataset_cleaner.utils_dataset import create_integrated_dataset

# --------- main -------------------
target_folder = (
    "/home/girobat/Projects/elliot/results/ml_100k_mobj_analysis/performance"
)
df_final = create_integrated_dataset(target_folder)
csv_path = Path(
    "/home/girobat/Projects/elliot/results/ml_100k_mobj_analysis/performance/final_knn_analysis.csv"
)
accuracy_metrics = ["nDCGRendle2020", "Recall", "HR", "Precision", "MAP", "MRR"]
beyond_accuracy = [
    "ItemCoverage",
    "UserCoverage",
    "ACLT",
    "Gini",
    "SEntropy",
    "EFD",
    "EPC",
    "PopREO",
    "PopRSP",
]
mobj_pipeline(csv_path, accuracy_metrics, beyond_accuracy, save=True)
