import numpy as np
import pandas as pd
import math

# ==============================================================================
# 1. SETUP YOUR EXPERIMENT HERE
# ==============================================================================

# Number of items (if ItemKNN) or Users (if UserKNN) in your dataset
DATASET_SIZE = 942

# The "Approximation Severity" thresholds you want to test.
TARGET_THRESHOLDS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Allowed deviation from the target threshold
TOLERANCE = 0.1

# Constraints
MAX_TABLES = 100
MAX_TREES_ANNOY = 200


# ==============================================================================
# 2. THEORETICAL SOLVERS
# ==============================================================================

def get_req_prob_cosine(sim_t):
    """Converts Target Cosine Similarity -> Required Collision Probability"""
    sim_t = min(1.0, max(-1.0, sim_t))
    return 1 - (math.acos(sim_t) / math.pi)


def get_sim_from_prob_cosine(prob):
    """Converts Actual Collision Probability -> Actual Cosine Similarity"""
    return math.cos((1 - prob) * math.pi)


def solve_s_curve(target_p, k_list, max_l):
    """
    Finds integer L for given k_list to match target probability P.
    """
    valid_configs = []

    for k in k_list:
        if target_p <= 0.0001:
            continue

        # Calculate ideal float L
        try:
            exact_L = (1 / target_p) ** k
        except OverflowError:
            continue

        # Check integers around the exact value
        candidates_L = [int(math.floor(exact_L)), int(math.ceil(exact_L))]
        candidates_L = sorted(list(set([l for l in candidates_L if 1 <= l <= max_l])))

        for L in candidates_L:
            # Re-calculate actual probability
            actual_p = (1 / L) ** (1 / k)
            valid_configs.append((k, L, actual_p))

    return valid_configs


# ==============================================================================
# 3. MAIN GENERATOR
# ==============================================================================

results = []

print(f"{'=' * 100}")
print(f"HYPERPARAMETER GENERATOR FOR DATASET N={DATASET_SIZE}")
print(f"Allowed Diff: ±{TOLERANCE}")
print(f"{'=' * 100}\n")

for target_t in TARGET_THRESHOLDS:

    # ---------------------------------------------------------
    # 1. MinHash (Jaccard)
    # Theory: Prob = Similarity
    # ---------------------------------------------------------
    req_p = target_t
    configs = solve_s_curve(req_p, k_list=[1, 2, 3, 4, 5, 6, 8, 10], max_l=MAX_TABLES)

    for k, L, act_p in configs:
        act_t = act_p
        diff = abs(act_t - target_t)
        if diff <= TOLERANCE:
            results.append({
                'Model': 'MinHash (Jaccard)',
                'Approximation Severity': target_t,
                'k value': k,
                'L value': L,
                'Actual t value': round(act_t, 5),
                'Diff': round(diff, 5)
            })

    # ---------------------------------------------------------
    # 2. LSH Random Projection & FairANN (Cosine)
    # Theory: Prob = 1 - arccos(Sim)/pi
    # ---------------------------------------------------------
    req_p = get_req_prob_cosine(target_t)
    configs = solve_s_curve(req_p, k_list=[1, 2, 4, 6, 8, 10, 12, 16], max_l=MAX_TABLES)

    for k, L, act_p in configs:
        act_t = get_sim_from_prob_cosine(act_p)
        diff = abs(act_t - target_t)
        if diff <= TOLERANCE:
            results.append({
                'Model': 'LSH-RP / FairANN',
                'Approximation Severity': target_t,
                'k value': k,
                'L value': L,
                'Actual t value': round(act_t, 5),
                'Diff': round(diff, 5)
            })

    # ---------------------------------------------------------
    # 3. FAISS LSH (Cosine)
    # Theory: Same as RP, but k = nbits
    # ---------------------------------------------------------
    configs = solve_s_curve(req_p, k_list=[2, 4, 8, 12, 16, 24, 32], max_l=MAX_TABLES)

    for k, L, act_p in configs:
        act_t = get_sim_from_prob_cosine(act_p)
        diff = abs(act_t - target_t)
        if diff <= TOLERANCE:
            results.append({
                'Model': 'FAISS (LSH)',
                'Approximation Severity': target_t,
                'k value': k,
                'L value': L,
                'Actual t value': round(act_t, 5),
                'Diff': round(diff, 5)
            })

    # ---------------------------------------------------------
    # 4. ANNOY (Cosine)
    # Theory: k is fixed by dataset size
    # ---------------------------------------------------------
    annoy_k = math.log2(DATASET_SIZE / 100)
    if annoy_k < 1: annoy_k = 1.0

    # Solve for L (n_trees)
    if req_p > 0.0001:
        exact_L = (1 / req_p) ** annoy_k
        candidates_L = [int(math.floor(exact_L)), int(math.ceil(exact_L))]
        candidates_L = sorted(list(set([l for l in candidates_L if 1 <= l <= MAX_TREES_ANNOY])))

        for L in candidates_L:
            act_p = (1 / L) ** (1 / annoy_k)
            act_t = get_sim_from_prob_cosine(act_p)
            diff = abs(act_t - target_t)

            if diff <= TOLERANCE:
                results.append({
                    'Model': 'ANNOY',
                    'Approximation Severity': target_t,
                    'k value': round(annoy_k, 2),  # Fixed k
                    'L value': L,
                    'Actual t value': round(act_t, 5),
                    'Diff': round(diff, 5)
                })

# ==============================================================================
# 4. PROCESSING, SORTING & EXPORT
# ==============================================================================

if not results:
    print("No configurations found within tolerance. Try increasing MAX_TABLES or TOLERANCE.")
else:
    df = pd.DataFrame(results)

    # 1. Define Custom Sort Order for Models
    custom_order = ['MinHash (Jaccard)', 'LSH-RP / FairANN', 'FAISS (LSH)', 'ANNOY']
    df['Model'] = pd.Categorical(df['Model'], categories=custom_order, ordered=True)

    # 2. Filter Best Config per Model/Threshold
    # Sort by Diff (asc), then L value (asc) to prefer efficiency/simplicity in ties
    df = df.sort_values(by=['Diff', 'L value'])

    # Group and pick top 1
    df = df.groupby(['Model', 'Approximation Severity'], observed=True).head(1)

    # 3. Final Sort
    df = df.sort_values(by=['Model', 'Approximation Severity'])

    # 4. Reorder Columns
    cols = ['Model', 'Approximation Severity', 'k value', 'L value', 'Actual t value', 'Diff']
    df = df[cols]

    # Display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(index=False))

    # Export
    filename = "user_hyperparameters.csv"
    df.to_csv(filename, index=False)
    print(f"\nSaved to {filename}")