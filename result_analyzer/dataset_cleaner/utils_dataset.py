import pandas as pd
import re
from pathlib import Path


def parse_model_string(model_str):
    """Parses the 'model' column into a Series of hyperparameters."""
    parts = model_str.split("_")
    res = {"Algorithm": parts[0]}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            res[k] = v
    return pd.Series(res)


def create_integrated_dataset(folder_path, output_filename="final_knn_analysis.csv"):
    # 1. Initialize Path object
    base_dir = Path(folder_path)
    output_path = Path(output_filename)
    # 2. Identify relevant KNN files using glob
    all_files = list(base_dir.glob("rec_*_cutoff_*_relthreshold_*.tsv"))

    all_dataframes = []

    for file_path in all_files:
        # 3. Extract metadata from the filename
        # 3. Extract metadata from the filename
        match = re.search(r"rec_(.+)_cutoff_(\d+)", file_path.name)

        if match:
            model_type_from_file = match.group(1)
            # cutoff_val = int(match.group(2))

            # 4. Read the TSV
            df = pd.read_csv(file_path, sep="\t")

            # 5. Clean internal 'model' column for hyperparameters
            params_df = df["model"].apply(parse_model_string)
            df_clean = pd.concat([params_df, df.drop(columns=["model"])], axis=1)

            # 6. Add context metadata
            df_clean["Filename_Model"] = model_type_from_file
            # df_clean["Cutoff"] = cutoff_val

            all_dataframes.append(df_clean)

    # 7. Combine and export to the source folder
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)

        # We join the base_dir with the output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_path, index=False)

        print(f"Successfully integrated {len(all_dataframes)} files.")
        print(f"Saved to: {output_path.absolute()}")
        return final_df

def add_cr_column(target_csv_path, verification_csv_path, output_path=None):
    """
    Merges CR values from verification_csv_path into target_csv_path based on model hyperparameters.
    """
    target_path = Path(target_csv_path)
    verif_path = Path(verification_csv_path)

    if not target_path.exists():
        print(f"Target file not found: {target_path}")
        return
    if not verif_path.exists():
        print(f"Verification file not found: {verif_path}")
        return

    df_target = pd.read_csv(target_path)
    df_verif = pd.read_csv(verif_path)

    # 1. Define Mapping: Verification Col -> Target Col
    col_mapping = {
        "model": "Filename_Model",
        "neighbors": "nn",
        "similarity": "sim",
        "n_tables": "nt",
        "n_hash": "h",
        "sampling_strategy": "strat",
        "n_trees": "tr",
        "nbits": "nb",
        "search_k": "k"
    }

    # 2. Rename columns in verification df
    relevant_cols = [c for c in col_mapping.keys() if c in df_verif.columns]
    rename_dict = {k: v for k, v in col_mapping.items() if k in relevant_cols}
    df_verif_renamed = df_verif.rename(columns=rename_dict)
    
    # --- FIX 1: Normalize Model Names ---
    # Map Verification names to Target names
    model_name_map = {
        "ItemANNFaissLSH": "ItemANNfaissLSH",
        "UserANNFaissLSH": "UserANNfaissLSH",
        "ItemAnnoy": "ItemANNOY",
        "UserAnnoy": "UserANNOY",
        "ItemKNNFairness": "ItemKNNfairness",
        "UserKNNFairness": "UserKNNfairness"
    }
    if "Filename_Model" in df_verif_renamed.columns:
        df_verif_renamed["Filename_Model"] = df_verif_renamed["Filename_Model"].replace(model_name_map)

    # --- FIX 2: Coalesce 't' into 'nt' in Target ---
    # Some models use 't' instead of 'nt'. We standardize on 'nt'.
    if "t" in df_target.columns:
        if "nt" in df_target.columns:
            # Fill missing 'nt' values with 't'
            df_target["nt"] = df_target["nt"].fillna(df_target["t"])
        else:
            # Rename 't' to 'nt'
            df_target = df_target.rename(columns={"t": "nt"})

    # --- FIX 3: Normalize Sampling Strategy Names ---
    # Map Target strategy names to Verification names
    strat_map = {
        "approx": "approx_degree",
        "no": "no_sampling"
    }
    if "strat" in df_verif_renamed.columns:
        # Verification uses full names, Target uses short names.
        # We should map Target short names to Verification full names to match.
        pass
    
    # Actually, simpler to map Target values to match Verification values
    if "strat" in df_target.columns:
         df_target["strat"] = df_target["strat"].replace(strat_map)
            
    # Re-evaluate common columns after fixes
    common_cols = list(set(df_target.columns).intersection(set(df_verif_renamed.columns)))
    
    if "CR" not in df_verif_renamed.columns:
        print("CR column not found in verification file.")
        return

    cols_to_use = common_cols + ["CR"]
    cols_to_use = list(set(cols_to_use))
    
    df_verif_ready = df_verif_renamed[cols_to_use].copy()

    # 3. Handle Data Types
    for col in common_cols:
        df_target[col] = pd.to_numeric(df_target[col], errors='ignore')
        df_verif_ready[col] = pd.to_numeric(df_verif_ready[col], errors='ignore')

        if df_target[col].dtype == object or df_verif_ready[col].dtype == object:
             df_target[col] = df_target[col].astype(str)
             df_verif_ready[col] = df_verif_ready[col].astype(str)
             
    # 4. Merge
    print(f"Merging on columns: {common_cols}")
    merged_df = pd.merge(df_target, df_verif_ready, on=common_cols, how="left")

    # 5. Save
    if output_path is None:
        final_output = target_path
    else:
        final_output = Path(output_path)
    
    final_output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(final_output, index=False)
    print(f"Successfully added CR column. Saved to: {final_output}")
    
    # Verification stats
    print(f"CR Match Rate: {merged_df['CR'].notna().mean():.2%}")
    print("Models with CR values:")
    print(merged_df[merged_df['CR'].notna()]['Filename_Model'].value_counts())
    
    print("\nModels missing CR values:")
    missing_cr = merged_df[merged_df['CR'].isna()]
    if not missing_cr.empty:
        print(missing_cr['Filename_Model'].value_counts())
    else:
        print("None")
