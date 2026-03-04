import os
import pandas as pd
import re

def main():
    base_dir = "results_annoy copy"

    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' not found. Please ensure the script is run from the correct directory.")
        return

    # Function to determine if a model should be kept
    def keep_model(model_name):
        """
        Returns True if we should keep the model, False if we should delete it.
        We delete if:
        It starts with 'UserANNOY' or 'ItemANNOY' AND it has '_s_k=' AND the value after '_s_k=' is NOT '-1'
        """
        if model_name.startswith("UserANNOY") or model_name.startswith("ItemANNOY"):
            match = re.search(r'_s_k=([-0-9.]+)', model_name)
            if match:
                s_k = match.group(1)
                # If s_k is not -1, we want to delete this one, so return False
                if s_k != '-1':
                    return False
        return True

    files_deleted = 0
    rows_deleted = 0
    modified_performance_files = 0

    print(f"Starting cleanup in '{base_dir}'...\n")

    for root, dirs, files in os.walk(base_dir):
        dirname = os.path.basename(root)
        
        # 1. Process "recs" directories
        if dirname == "recs":
            for file in files:
                if not file.endswith(".tsv"):
                    continue
                
                # The filename itself usually contains the model name
                model_name = file.replace(".tsv", "")
                
                # check if it starts with rec_ like in performance
                if model_name.startswith("rec_"):
                    model_name = model_name[4:]

                if not keep_model(model_name):
                    filepath = os.path.join(root, file)
                    print(f"Deleting recs file: {filepath}")
                    os.remove(filepath)
                    files_deleted += 1

        # 2. Process "performance" directories
        if dirname == "performance":
            for file in files:
                if not file.endswith(".tsv"):
                    continue
                
                filepath = os.path.join(root, file)
                try:
                    df = pd.read_csv(filepath, sep='\t')
                    
                    if 'model' not in df.columns:
                        continue
                    
                    original_len = len(df)
                    
                    # Filter rows
                    df_filtered = df[df['model'].apply(lambda x: keep_model(str(x)))]
                    new_len = len(df_filtered)
                    
                    if new_len < original_len:
                        deleted_count = original_len - new_len
                        print(f"Updating performance file: {filepath} ({deleted_count} rows removed)")
                        df_filtered.to_csv(filepath, sep='\t', index=False)
                        modified_performance_files += 1
                        rows_deleted += deleted_count

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    print("\n--- Cleanup Summary ---")
    print(f"Recs files deleted: {files_deleted}")
    print(f"Performance files modified: {modified_performance_files}")
    print(f"Total performance rows deleted: {rows_deleted}")
    print("Cleanup completed successfully.")

if __name__ == "__main__":
    main()
