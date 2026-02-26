# `create_datasets.py` Usage Guide

## What it does
`create_datasets.py` is a utility script that aggregates and processes Elliot recommendation framework result files into a single consolidated dataset. Specifically, it performs two main tasks:
1. **Integration**: It reads all the individual performance metrics files from a specified results directory (e.g., `results/.../performance`) and compiles them into a single integrated CSV dataset inside the `data/` directory.
2. **Component Return (CR) Verification**: It reads a secondary verification file (`CR_verification.csv`) containing computed Component Return values and merges these CR values into the newly integrated dataset based on matching model hyperparameters.

## How to use it
1. Open up `create_datasets.py` in your text editor.
2. Locate the `target_t_path` variable near the top of the script (around line 3):
   ```python
   target_t_path = 'target_t_0_05'
   ```
3. Change `target_t_path` to the name of the folder inside `results/` that you want to process.
4. Run the script from the terminal:
   ```bash
   python create_datasets.py
   ```

## Output
- **Console Output**: A brief progress summary, followed by a preview of the first 5 rows showing model filenames and their respective `CR` values, as well as the total number of processed rows.
- **Dataset File**: A single `.csv` file saved to the `data/` directory (e.g., `data/target_t_0_05_annoy.csv`). This file will contain all the performance metrics along with the merged `CR` column.
