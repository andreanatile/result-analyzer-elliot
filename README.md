# Information Retrieval: Evaluating ANN Models vs Exact kNN

## 🎯 Project Goals
The main objective of this project is to evaluate the trade-offs between exact k-Nearest Neighbors (kNN) and Approximate Nearest Neighbor (ANN) models in Recommender Systems. Specifically, the project focuses on:
* **Theoretical study** of the kNN baseline.
* **Analysis and implementation** of various ANN models (LSH, Faiss, Annoy, FairANN).
* **Evaluation of Approximation Severity**, providing a standardized way to compare different models .
* **Multi-Objective Optimization Analysis** using Pareto frontiers and Hypervolume to evaluate models across multiple dimensions simultaneously.
* **Development of an extensive visualization framework** to analyze the results.

---

## 🧠 Background

### The kNN Baseline
Neighbor-based Collaborative Filtering relies on the premise that users who agreed on items in the past will agree in the future (User-based) or that items bought by similar sets of users are similar (Item-based).
While exact kNN is considered the "highly accurate – gold standard," it is extremely computationally expensive ($O(N^2)$), as it requires computing the similarity of an item with every other item in the catalog .

### ANN Models Investigated
To solve the computational bottleneck of exact kNN, we implemented Approximate Nearest Neighbor (ANN) models that group items into "buckets" to limit comparisons. The models analyzed include:
* **LSH (Locality-Sensitive Hashing):** Uses MinHashing for Jaccard similarity and Random Projection for Cosine similarity .
* **Faiss:** Developed by Meta, it clusters space into Voronoi Cells to search only within the closest centroid to a query.
* **Annoy:** Developed by Spotify, it slices space with random hyperplanes to build forests of binary trees.
* **FairANN:** Stems from standard LSH but applies a specific sampling strategy to achieve fairness.

---

## 🎛️ Core Concept: Approximation Severity ($t$)
To fairly compare different ANN models, we utilize the concept of **Approximation Severity**, defined by a threshold ($t$) . 

* **The S-Curve:** The threshold $t$ models the probability of collision—how similar two items must be to end up in the same bucket. 
* **Control Parameters:** The threshold is controlled by the hyperparameter $k$ (AND matching, makes the model stricter) and $L$ (OR matching, makes the model looser) . 
* **Model Configurations:** A higher threshold results in a strict approximation, while a lower threshold yields a loose approximation . The exact mapping of $k$ and $L$ to $t$ depends on whether the underlying similarity metric is Jaccard ($t \approx (1/L)^{1/k}$) or Cosine ($P = 1 - (\arccos(t) / \pi)$).

---

## 🧪 Experimental Setup
All experiments were conducted with the following configuration:
* **Dataset:** Movielens100k .
* **Setup:** `user_k_core`: 20, `Top k`: 50, `Cutoff`: 20 .
* **Neighborhood sizes tested:** 50, 100, and 250 .
* **Similarities:** Jaccard and Cosine (Angular for Annoy) .
* **Approximation Severity levels evaluated:** [0.05, 0.25, 0.50, 0.75, 0.95] .

---

## 📊 Research Questions & Results

### RQ1: The Accuracy-Efficiency Trade-off
**Question:** How does approximation severity impact recommendation accuracy and computational efficiency? 

* **Accuracy:** Introducing approximation predictably causes a drop in accuracy compared to the kNN baseline . However, **Annoy** consistently maintained excellent accuracy, outperforming other ANN models . FairANN performed best at a "sweet spot" of $t \approx 0.75$, suffering at extremes due to noise or excessive strictness .
* **Efficiency:** Efficiency was measured using the **Candidate Ratio (CR)**, which computes how much the search space was pruned ($CR \ll 1$) . ANN models were significantly more efficient, running up to **10 times faster** than exact kNN . FairANN and Annoy demonstrated the highest overall efficiency .

### RQ2: Beyond-Accuracy Metrics (Diversity, Novelty, Fairness)
**Question:** How does approximation impact non-traditional metrics? 

* **Fairness (PopREO):** Exact kNN models suffer heavily from popularity bias. The inherent noise introduced by ANN approximations actually makes recommendations *more fair*, filtering out some of the popularity bias.
* **Diversity (Gini Index):** As approximation strictness rises, diversity improves for User-based models but drops for Item-based models .
* **Novelty (EPC):** Generally, approximation inhibits novelty because it focuses the search on denser areas of the vector space. **Annoy** is a notable exception; its intrinsic randomness allows it to maintain high novelty .
* **Long-Tail (APLT):** Annoy showcased the best overall fairness for long-tail items, while FairANN peaked at the $0.75$ threshold .

### RQ3: Multi-Objective Optimization (Pareto Dominance)
**Question:** How do the models compare when evaluating multiple objectives simultaneously? 

Hypervolume measures were used for evaluating difference performance across conflicting objectives . We tested four scenarios:
1. **Content Delivery (nDCG, Gini, EPC):** Annoy displays the best diversity and novelty, surpassing even kNN models .
2. **Efficiency (nDCG, CR):** Annoy offers the highest temporal efficiency while maintaining optimal accuracy .
3. **Content Exposure (nDCG, APLT):** FairANN models offered the best fairness due to their specific sampling strategies .
4. **Mix (nDCG, PopREO, CR):** All ANN models provided a highly balanced trade-off compared to exact kNN .

---

## 🚀 Conclusion
While ANN models sacrifice a small quota of accuracy compared to exact kNN, they gain a **massive advantage in temporal efficiency** . Furthermore, this project demonstrates that absolute accuracy is not the only valuable metric in recommendation systems. In real-world scenarios, the trade-off of introducing approximation yields **highly appreciated improvements in beyond-accuracy metrics** like novelty, catalog discovery, and provider fairness . 
