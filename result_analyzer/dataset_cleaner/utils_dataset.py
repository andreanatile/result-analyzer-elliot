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
    else:
        print(f"No matching files found in {base_dir.absolute()}")
        return None
