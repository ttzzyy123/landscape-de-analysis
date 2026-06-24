Step 10D final supervisor-requested tables under 0307 main tiebreak scheme

Generated files:
- Table 1 CSV: output/0307_main_comparison/table01_bbob_group_hyperparameter_shap_distance_0307.csv
- Table 1 MD:  output/0307_main_comparison/table01_bbob_group_hyperparameter_shap_distance_0307.md
- Table 2 CSV: output/0307_main_comparison/table02_generated_predicted_vs_actual_hof_0307.csv
- Table 2 MD:  output/0307_main_comparison/table02_generated_predicted_vs_actual_hof_0307.md
- Table 3 CSV: output/0307_main_comparison/table03_generated_shap_shape_distance_0307.csv
- Table 3 MD:  output/0307_main_comparison/table03_generated_shap_shape_distance_0307.md
- Table 4 CSV: output/0307_main_comparison/table04_generated_shap_value_distance_0307.csv
- Table 4 MD:  output/0307_main_comparison/table04_generated_shap_value_distance_0307.md
- Table 5 CSV: output/0307_main_comparison/table05_generated_shap_similarity_summary_0307.csv
- Table 5 MD:  output/0307_main_comparison/table05_generated_shap_similarity_summary_0307.md

Inputs:
- Hall of Fame: output/0307_main_comparison/hall_of_fame_0307.csv
- SHAP similarity: output/0307_main_comparison/shap_similarity_0307.csv
- Group SHAP dir: output/shap_eps2bins_3_r0307_main_tiebreak_niles
- BBOB individual SHAP dir: output/shap_individual_manual_bins_niles
- Affine SHAP dir: output/new_function_task/task_H_individual_shap_affine_functions
- LLaMEA SHAP dir: output/new_function_task/task_H_individual_shap_real_generated_function

Definitions:
- Table 1 distance = abs(group normalized SHAP importance - BBOB function normalized SHAP importance).
- Table 2 predicted configuration = the BBOB Hall-of-Fame winner inside the assigned group.
- Table 2 actual configuration = the generated function's own Hall-of-Fame winner.
- Table 3 shape distance = Wasserstein distance and KS statistic between group and function SHAP distributions.
- Table 4 value distance = differences in mean SHAP, mean absolute SHAP, and normalized SHAP importance.
- Table 5 summary combines cosine, Spearman, top-3 overlap, mean value distance, and mean shape distance.

Important limitation:
- If raw SHAP-value files are unavailable, Tables 3 and 4 fall back to summary-level SHAP importance values.
- In fallback mode, shape-distance results should be interpreted as summary-level approximations, not full raw SHAP distribution comparisons.
