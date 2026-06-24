Step 10 SHAP similarity analysis under the 0307 main grouping scheme

Scheme used:
- eps2bins_3_r0307_main_tiebreak

This script intentionally uses:
- output/0307_main_comparison/hall_of_fame_0307.csv
- output/shap_eps2bins_3_r0307_main_tiebreak_niles
- output/shap_individual_manual_bins_niles
- output/new_function_task/task_H_individual_shap_affine_functions
- output/new_function_task/task_H_individual_shap_real_generated_function

It does not use ambiguous_on_tie, mean_based, eps2bins_3_r025065, or encoding SHAP outputs.

Method:
- Read Hall of Fame rows and remove aggregate All rows.
- For each function, read assigned_group_0307.
- Load the corresponding group-level SHAP reference shap_group_k.csv.
- Load the individual SHAP importance file for BBOB, affine, or LLaMEA functions.
- Convert both into normalized feature-importance vectors.
- Compute cosine similarity, Spearman rank correlation, L1/L2 distance, top-3 overlap, largest per-feature difference, and Wasserstein-style summary metrics.

Wasserstein note:
- True per-feature Wasserstein over raw SHAP distributions requires raw SHAP values.
- Current inputs are summary-level importance CSV files.
- Therefore mean_wasserstein / median_wasserstein / max_wasserstein are computed from per-feature absolute differences.
- wasserstein_importance_distribution compares the distribution of normalized importance magnitudes across features.

Outputs:
- output/0307_main_comparison/shap_similarity_0307.csv
- output/0307_main_comparison/shap_similarity_0307.md
- output/0307_main_comparison/shap_similarity_0307_by_function_type.csv
- output/0307_main_comparison/shap_similarity_0307_by_group.csv
