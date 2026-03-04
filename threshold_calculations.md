# Threshold ($t$) Calculation Explained

The `t_hyperparams_finder.py` script maps an "Approximation Severity" threshold ($t$) to the underlying hyperparameters of different Approximate Nearest Neighbor (ANN) algorithms. Specifically, it searches for the optimal hash length ($k$) and number of tables/trees ($L$) to approximate the target similarity threshold.

Below are the exact mathematical differences in how the target $t$ is translated into $k$ and $L$ for each model, and how the *actual* resulting threshold ($t_{actual}$) is calculated backward from the selected integer hyper-parameters.

## 1. MinHash (Jaccard Similarity)
For MinHash, the probability of a hash collision is exactly equal to the Jaccard similarity. The relationship relies purely on the standard LSH S-curve equation.

*   **Target Probability:** $P = t$
*   **Finding $L$:** $L = \left(\frac{1}{t}\right)^k$
*   **Calculating Actual Threshold ($t_{actual}$):**
    Once integer values for $L$ and $k$ are chosen, the actual threshold achieved is computed directly from the resulting probability:
    $$t_{actual} = \left(\frac{1}{L}\right)^{\frac{1}{k}}$$

## 2. FairANN, LSH-RP, and FAISS LSH (Cosine Similarity)
Cosine similarity methods use Random Projection. Here, the collision probability is a function of the angle between vectors, not the direct similarity value.

*   **Target Probability:** The similarity $t$ must first be converted into a collision probability.
    $$P_{target} = 1 - \frac{\arccos(t)}{\pi}$$
*   **Finding $L$:** $L = \left(\frac{1}{P_{target}}\right)^k$
*   **Calculating Actual Threshold ($t_{actual}$):**
    Once integer values for $L$ and $k$ are chosen, the script calculates the actual collision probability and converts it *back* to Cosine similarity:
    $$P_{actual} = \left(\frac{1}{L}\right)^{\frac{1}{k}}$$
    $$t_{actual} = \cos((1 - P_{actual}) \cdot \pi)$$

## 3. ANNOY (Cosine Similarity)
ANNOY uses the same Random Projection principles (and thus the same Cosine-to-Probability conversions) as FairANN and FAISS, but with a major constraint: **$k$ is fixed based on the dataset size ($N$)**.

*   **Fixed $k$:** $k = \max\left(1.0, \log_2\left(\frac{N}{100}\right)\right)$
    *(For your dataset of $N=942$, $k \approx 3.23$)*
*   **Target Probability:**
    $$P_{target} = 1 - \frac{\arccos(t)}{\pi}$$
*   **Finding $L$ (Number of Trees):** Since $k$ is fixed, the script uniquely solves for $L$.
    $$L = \left(\frac{1}{P_{target}}\right)^k$$
*   **Calculating Actual Threshold ($t_{actual}$):**
    $$P_{actual} = \left(\frac{1}{L}\right)^{\frac{1}{k}}$$
    $$t_{actual} = \cos((1 - P_{actual}) \cdot \pi)$$

---
### Summary of the Core Differences
*   **MinHash** operates directly in **Probability = Similarity** space.
*   **LSH-RP / FAISS / FairANN** map back and forth between **Cosine Similarity $\leftrightarrow$ Angle $\leftrightarrow$ Probability** space to evaluate the LSH parameters.
*   **ANNOY** relies on the same angle/probability translation as Cosine methods, but uniquely enforces that **$k$ is a strict function of the dataset size**. This fundamentally limits the shapes of the S-curves it can draw compared to FAISS/LSH-RP.
