import numpy as np
import pandas as pd
import math

# ==============================================================================
# 1. SETUP YOUR EXPERIMENT HERE
# ==============================================================================

# Movielens100k after preprocessing user 942 items 1348
#  

DATASET_SIZE = 1348
 

# The "Approximation Severity" thresholds you want to test (t).
# t = 0.9: Strict (Fast, High Approx Error, Low Recall) 
# t = 0.2: Loose (Slow, Low Approx Error, High Recall)
TARGET_THRESHOLDS = [0.2, 0.4, 0.6, 0.8]

# Allowed deviation from the target threshold (e.g., target 0.5 -> accept 0.45 to 0.55)
TOLERANCE = 0.05 

# Constraints to prevent generating unusable configs (e.g., 1000 tables = RAM crash)
MAX_TABLES = 100
MAX_TREES_ANNOY = 200

# ==============================================================================
# 2. THEORETICAL MATH FUNCTIONS
# ==============================================================================

def get_req_prob_cosine(sim_t):
    """
    Converts Target Cosine Similarity (t) to Required Collision Probability (p).
    Theory: For Random Projection, P = 1 - theta/pi
    """
    sim_t = min(1.0, max(-1.0, sim_t))
    return 1 - (math.acos(sim_t) / math.pi)

def get_sim_from_prob_cosine(prob):
    """
    Reverse: Converts Actual Collision Probability (p) to Actual Cosine Similarity.
    """
    return math.cos((1 - prob) * math.pi)

def solve_s_curve(target_p, k_list, max_l):
    """
    Finds integer L (Tables) for given k (Hashes) to match target probability P.
    Equation: P = 1 - (1 - p^k)^L  -> Approximated at inflection point as P_thresh = (1/L)^(1/k)
    Therefore: L = (1/P)^k
    """
    valid_configs = []
    
    for k in k_list:
        if target_p <= 0.001: continue
        
        # Calculate ideal float L
        exact_L = (1 / target_p) ** k
        
        # Check integer neighbors (floor and ceil)
        candidates_L = [int(math.floor(exact_L)), int(math.ceil(exact_L))]
        candidates_L = sorted(list(set([l for l in candidates_L if 1 <= l <= max_l])))
        
        for L in candidates_L:
            # Re-calculate actual probability resulting from these integers
            actual_p = (1 / L) ** (1 / k)
            valid_configs.append((k, L, actual_p))
            
    return valid_configs

# ==============================================================================
# 3. MAIN GENERATOR LOOP
# ==============================================================================

results = []

print(f"{'='*100}")
print(f"HYPERPARAMETER GENERATOR FOR DATASET N={DATASET_SIZE}")
print(f"Aligning all models to Unified Severity Threshold t (S-Curve)")
print(f"{'='*100}\n")

for target_t in TARGET_THRESHOLDS:
    
    # --- MODEL 1: MinHash (Jaccard) ---
    # Theory: Prob = Similarity (No correction needed)
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

    # --- MODEL 2: LSH Random Projection & FairANN (Cosine) ---
    # Theory: Prob = 1 - arccos(Sim)/pi (Cosine Correction)
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

    # --- MODEL 3: FAISS LSH (Cosine) ---
    # Theory: Same as RP, but k = nbits. 
    # Note: We create 'n_tables' indices to simulate L.
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

    # --- MODEL 4: ANNOY (Cosine) ---
    # Theory: k is fixed by dataset size (Depth). We only tune L (n_trees).
    # k approx log2(N / 100)
    annoy_k = math.log2(DATASET_SIZE / 100) 
    
    # Solve for L (n_trees) -> L = (1/P)^k
    if req_p > 0.001:
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
# 4. PRINT RESULTS
# ==============================================================================
if not results:
    print("No configurations found within tolerance. Try increasing MAX_TABLES or TOLERANCE.")
else:
    df = pd.DataFrame(results)
    
    # Sort by Target, then Model, then by smallest Difference (Best Accuracy)
    df = df.sort_values(by=['Target t', 'Model', 'Diff'])
    df.to_csv('data/ann_configs.csv', index=False)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Print clean table for copy-pasting
    print(df[['Target t', 'Model', 'Elliot Config', 'Actual t', 'Diff']].to_string(index=False))