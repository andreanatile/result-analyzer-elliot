from result_analyzer.dataset_cleaner.utils_dataset import create_integrated_dataset, add_cr_column

target_t_path = 'target_t_0_95_minh'

path = f"results/{target_t_path}/performance"
target_file = f"data/{target_t_path}.csv"
verification_file = f"results/{target_t_path}/performance/CR_verification.csv"

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
