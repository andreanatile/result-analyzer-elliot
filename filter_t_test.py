import pandas as pd
import re

def analyze_targeted_ttest(input_file, output_file='filtered_ttest_ANN_vs_KNN.csv', alpha=0.05):
    print(f"Reading file: {input_file}")
    
    # Read the TSV. We assign the column names since there is no header
    df = pd.read_csv(input_file, sep='\t', header=None, 
                     names=['Model_A', 'Model_B', 'Metric', 'p_value'])

    # Advanced function to extract info from the model name
    def extract_info(model_string):
        # The base name is the first word before the underscore (e.g., ItemANNOY, ItemKNNfairness, ItemKNN)
        base = model_string.split('_')[0]
        
        # Model type (User or Item)
        model_type = 'User' if model_string.startswith('User') else 'Item'
        
        # Extract the 'nn' value
        match = re.search(r'nn=(\d+)', model_string)
        nn_val = int(match.group(1)) if match else None
        
        # Determine if it is the absolute BASELINE (exactly ItemKNN or UserKNN)
        is_baseline = (base == 'ItemKNN' or base == 'UserKNN')
        
        return pd.Series([base, model_type, nn_val, is_baseline])

    print("Extracting information from model names...")
    df[['Base_A', 'Type_A', 'nn_A', 'is_baseline_A']] = df['Model_A'].apply(extract_info)
    df[['Base_B', 'Type_B', 'nn_B', 'is_baseline_B']] = df['Model_B'].apply(extract_info)

    # --- FILTERS ---
    
    # 1. We want Model_A to be the target model (ANN, FairANN, KNNfairness) 
    #    and Model_B to be the pure BASELINE (ItemKNN or UserKNN).
    #    This also removes symmetric duplicates (B vs A).
    mask_A_is_target = ~df['is_baseline_A']
    mask_B_is_baseline = df['is_baseline_B']
    
    # 2. We want to compare only Item vs Item or User vs User
    mask_same_type = df['Type_A'] == df['Type_B']
    
    # 3. We want to compare only models with the same number of neighbors (nn)
    mask_same_nn = df['nn_A'] == df['nn_B']
    
    # Apply all filters simultaneously
    filtered_df = df[mask_A_is_target & mask_B_is_baseline & mask_same_type & mask_same_nn].copy()

    # --- STATISTICAL EVALUATION ---
    # The difference is significant if p-value < alpha (0.05)
    filtered_df['Significant_Difference'] = filtered_df['p_value'] < alpha

    # Drop temporary support columns
    filtered_df = filtered_df.drop(columns=[
        'Base_A', 'Type_A', 'nn_A', 'is_baseline_A',
        'Base_B', 'Type_B', 'nn_B', 'is_baseline_B'
    ])

    # Sort the results for easier reading: 
    # Baseline first, then Metric, then Tested Model
    filtered_df = filtered_df.sort_values(by=['Model_B', 'Metric', 'Model_A'])

    # Save to CSV
    filtered_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Filtering completed successfully!")
    print(f"Total original rows: {len(df)}")
    print(f"Valid comparisons found (ANN/Fair vs Baseline, same nn, same Type): {len(filtered_df)}")
    print(f"File saved to: {output_file}")

    # Brief summary
    significant_count = filtered_df['Significant_Difference'].sum()
    print(f"Out of these valid comparisons, {significant_count} show a statistically significant difference (p < {alpha}).")

import os
import glob

result_dir = "result_t_test"
output_dir = "filtered_t_test"

# Crea la cartella di output se non esiste
os.makedirs(output_dir, exist_ok=True)

# Trova tutti i file .tsv nella cartella result_t_test
tsv_files = glob.glob(os.path.join(result_dir, "*.tsv"))

if not tsv_files:
    print(f"Nessun file TSV trovato nella cartella: {result_dir}")
else:
    print(f"Trovati {len(tsv_files)} file TSV da elaborare in {result_dir}.")
    
    for tsv_file in sorted(tsv_files):
        # Estrai il nome del file senza estensione
        base_name = os.path.basename(tsv_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Crea un nome di file di output dinamico basato sul nome di input nella nuova cartella
        output_filename = os.path.join(output_dir, f"filtered_{name_without_ext}.csv")
        
        print(f"\n{'='*60}")
        print(f"Elaborazione di: {base_name}")
        print(f"{'='*60}")
        
        try:
            analyze_targeted_ttest(tsv_file, output_file=output_filename)
        except Exception as e:
            print(f"Errore durante l'elaborazione di {tsv_file}: {e}")