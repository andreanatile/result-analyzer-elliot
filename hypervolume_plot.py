import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse


def rename_algorithm(row):
    """
    Custom logic to rename specific FairANN baselines to their
    theoretical algorithm names based on similarity and strategy.
    """
    # 1. Safely extract values (handling both 'sim'/'strat' and the older 'Similarity'/'Algorithm' formats)
    algo = str(row.get('Algorithm', ''))
    strat = str(row.get('strat', algo))  # Fallback to algo string if 'strat' column doesn't exist
    sim_col_name = 'sim' if 'sim' in row else 'Similarity'
    sim = str(row.get(sim_col_name, '')).lower()

    # Check if the model is a FairANN variant and uses 'no_sampling'
    is_fair_ann = ('UserFairANN' in algo) or ('ItemFairANN' in algo)
    is_no_sampling = 'no_sampling' in strat

    # Apply renaming rules
    if is_fair_ann and is_no_sampling:
        if sim == 'jaccard':
            return 'Minhashing'
        elif sim == 'cosine':
            return 'LSH Random Projection'

    # If no rules match, return the default Algorithm_Variant (or Algorithm)
    return row.get('Algorithm_Variant', algo)


def create_hypervolume_charts(csv_file):
    """
    Reads a CSV file containing Hypervolume results and generates
    two descending bar charts (one for User models, one for Item models).
    """
    print(f"Processing file: {csv_file}")

    # Read the dataframe
    df = pd.read_csv(csv_file)

    # Apply the custom renaming logic to create the final display names for the X-axis
    df['Display_Name'] = df.apply(rename_algorithm, axis=1)
    x_axis = 'Display_Name'

    # The two types of models to analyze
    model_types = ['User', 'Item']

    for model_type in model_types:
        # 1. Filter the dataframe for the current type (User or Item)
        filtered_df = df[df['Type'] == model_type].copy()

        # If there are no models of this type in the file, skip to the next
        if filtered_df.empty:
            print(f"No models of type '{model_type}' found in the file.")
            continue

        # 2. Sort in descending order based on Hypervolume
        sorted_df = filtered_df.sort_values(by='Hypervolume', ascending=False)

        # 3. Chart creation
        plt.figure(figsize=(12, 6))  # Set a good size for the chart

        # Draw the bar chart using Seaborn
        ax = sns.barplot(
            data=sorted_df,
            x=x_axis,
            y='Hypervolume',
            palette='viridis'  # An elegant and scientific color palette
        )

        # 4. Aesthetic customization (Titles, Axes, etc.)
        base_filename = os.path.basename(csv_file).replace('.csv', '')
        clean_title_name = base_filename.replace('_', ' ').title()

        plt.title(f'Hypervolume Comparison - {model_type} Models\n({clean_title_name})', fontsize=16, fontweight='bold')
        plt.xlabel('Model Name', fontsize=12)
        plt.ylabel('Hypervolume Value', fontsize=12)

        # Rotate X-axis labels by 45 degrees so they don't overlap
        plt.xticks(rotation=45, ha='right')

        # Add a horizontal grid to make values easier to read
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Optimize spacing so labels aren't cut off
        plt.tight_layout()

        # 5. Save the chart in the same directory as the CSV file
        output_dir = os.path.dirname(csv_file)
        if output_dir == '':
            output_dir = '.'  # Current directory if no path is provided

        output_name = os.path.join(output_dir, f"chart_{base_filename}_{model_type.lower()}.png")
        plt.savefig(output_name, dpi=300)
        print(f"-> Chart saved: {output_name}")

        # Close the plot to free up memory
        plt.close()


# ==========================================
# COMMAND LINE INTERFACE SETUP
# ==========================================
if __name__ == "__main__":
    csv_path = './hypervolume_results/efficency.csv'

    if os.path.isfile(csv_path):
        create_hypervolume_charts(csv_path)
    else:
        print(f"Error: The file '{csv_path}' does not exist.")
        print("Please check the path and try again.")
