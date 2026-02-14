from result_analyzer.dataset_cleaner.utils_dataset import create_integrated_dataset, add_cr_column

path="results/target_t_0_05/performance"
target_file = "data/target_t_0_05.csv"
verification_file = "results/target_t_0_05/performance/CR_verification.csv"

# 1. Create dataset
print("Creating dataset...")
create_integrated_dataset(path, target_file)

# 2. Add CR column
print("\nAdding CR column...")
add_cr_column(target_file, verification_file, target_file)

import pandas as pd
df = pd.read_csv(target_file)
print("\nFirst 5 rows with CR:")
print(df[['Filename_Model', 'CR']].head())
print(f"\nTotal rows: {len(df)}")
print(f"Rows with CR: {df['CR'].notna().sum()}")
