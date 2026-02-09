import numpy as np
import pandas as pd
import math

# ==============================================================================
# 1. SETUP YOUR EXPERIMENT HERE
# ==============================================================================

# Number of items (if ItemKNN) or Users (if UserKNN) in your dataset
# Example: MovieLens 1M has ~3,706 items. 
DATASET_SIZE = 942

# The "Approximation Severity" thresholds you want to test.
# 0.9 = Strict (Fast, Low Recall) | 0.2 = Loose (Slow, High Recall)
TARGET_THRESHOLDS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Allowed deviation from the target threshold
TOLERANCE = 0.1

# Constraints to prevent generating unusable configs (e.g., 1000 tables)
MAX_TABLES = 100
MAX_TREES_ANNOY = 200

# ==============================================================================
# 2. THEORETICAL SOLVERS
# ==============================================================================

def get_req_prob_cosine(sim_t):
    """Converts Target Cosine Similarity -> Required Collision Probability"""
    # Formula: P = 1 - theta/pi
    sim_t = min(1.0, max(-1.0, sim_t))
    return 1 - (math.acos(sim_t) / math.pi)

def get_sim_from_prob_cosine(prob):
    """Converts Actual Collision Probability -> Actual Cosine Similarity"""
    # Formula: Sim = cos((1-P) * pi)
    return math.cos((1 - prob) * math.pi)

def solve_s_curve(target_p, k_list, max_l):
    """
    Finds integer L for given k_list to match target probability P.
    Equation: P = (1/L)^(1/k)  =>  L = (1/P)^k
    """
    valid_configs = []
    
    for k in k_list:
        if target_p <= 0.001: continue
        
        # Calculate ideal float L
        exact_L = (1 / target_p) ** k
        
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

print(f"{'='*100}")
print(f"HYPERPARAMETER GENERATOR FOR DATASET N={DATASET_SIZE}")
print(f"Allowed Tolerance: ±{TOLERANCE}")
print(f"{'='*100}\n")

for target_t in TARGET_THRESHOLDS:
    print(f"Processing Target Threshold t = {target_t} ...")
    
    # ---------------------------------------------------------
    # MODEL A: MinHash (Jaccard)
    # Theory: Prob = Similarity
    # ---------------------------------------------------------
    req_p = target_t
    configs = solve_s_curve(req_p, k_list=[2, 3, 4, 5, 6, 8, 10], max_l=MAX_TABLES)
    
    for k, L, act_p in configs:
        act_t = act_p # For Jaccard, T = P
        if abs(act_t - target_t) <= TOLERANCE:
            results.append({
                'Target t': target_t,
                'Model': 'MinHash (Jaccard)',
                'Param k (AND)': k,
                'Param L (OR)': L,
                'Actual t': round(act_t, 4),
                'Diff': round(abs(act_t - target_t), 4),
                'Elliot Config': f"n_hash: {k}, n_tables: {L}"
            })

    # ---------------------------------------------------------
    # MODEL B: LSH Random Projection & FairANN (Cosine)
    # Theory: Prob = 1 - arccos(Sim)/pi
    # ---------------------------------------------------------
    req_p = get_req_prob_cosine(target_t)
    configs = solve_s_curve(req_p, k_list=[2, 4, 6, 8, 10, 12, 16], max_l=MAX_TABLES)
    
    for k, L, act_p in configs:
        act_t = get_sim_from_prob_cosine(act_p)
        if abs(act_t - target_t) <= TOLERANCE:
            results.append({
                'Target t': target_t,
                'Model': 'LSH-RP / FairANN',
                'Param k (AND)': k,
                'Param L (OR)': L,
                'Actual t': round(act_t, 4),
                'Diff': round(abs(act_t - target_t), 4),
                'Elliot Config': f"n_hash: {k}, n_tables: {L}"
            })

    # ---------------------------------------------------------
    # MODEL C: FAISS LSH (Cosine)
    # Theory: Same as RP, but k = nbits
    # ---------------------------------------------------------
    # We restrict nbits to standard sizes or small multiples
    configs = solve_s_curve(req_p, k_list=[4, 8, 12, 16, 24, 32], max_l=MAX_TABLES)
    
    for k, L, act_p in configs:
        act_t = get_sim_from_prob_cosine(act_p)
        if abs(act_t - target_t) <= TOLERANCE:
            results.append({
                'Target t': target_t,
                'Model': 'FAISS (LSH)',
                'Param k (AND)': k,
                'Param L (OR)': L,
                'Actual t': round(act_t, 4),
                'Diff': round(abs(act_t - target_t), 4),
                'Elliot Config': f"nbits: {k}, n_tables: {L}"
            })

    # ---------------------------------------------------------
    # MODEL D: ANNOY (Cosine)
    # Theory: k is fixed by dataset size
    # ---------------------------------------------------------
    # Calculate implicit depth k
    annoy_k = math.log2(DATASET_SIZE / 100) # Assuming leaf size 100
    
    # Solve for L (n_trees)
    # L = (1/P)^k
    if req_p > 0:
        exact_L = (1 / req_p) ** annoy_k
        candidates_L = [int(math.floor(exact_L)), int(math.ceil(exact_L))]
        candidates_L = sorted(list(set([l for l in candidates_L if 1 <= l <= MAX_TREES_ANNOY])))
        
        for L in candidates_L:
            act_p = (1 / L) ** (1 / annoy_k)
            act_t = get_sim_from_prob_cosine(act_p)
            
            if abs(act_t - target_t) <= TOLERANCE:
                 results.append({
                    'Target t': target_t,
                    'Model': 'ANNOY',
                    'Param k (AND)': round(annoy_k, 2),
                    'Param L (OR)': L,
                    'Actual t': round(act_t, 4),
                    'Diff': round(abs(act_t - target_t), 4),
                    'Elliot Config': f"n_trees: {L}"
                })

# ==============================================================================
# 4. DISPLAY RESULTS
# ==============================================================================
if not results:
    print("No configurations found within tolerance. Try increasing MAX_TABLES or TOLERANCE.")
else:
    df = pd.DataFrame(results)
    
    # Sort by Target, then Model, then by smallest Difference (Accuracy)
    df = df.sort_values(by=['Target t', 'Model', 'Diff'])
    
    # Filter to show only the BEST configuration per Model per Target?
    # Uncomment the line below if you only want the Single Best config per model
    df = df.loc[df.groupby(['Target t', 'Model'])['Diff'].idxmin()]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(df[['Target t', 'Model', 'Elliot Config', 'Actual t', 'Diff']].to_string(index=False))

    # Optional: Export to CSV
    df.to_csv("user_hyperparameters.csv", index=False)