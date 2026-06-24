Step 10B missing table generation under 0307 main tiebreak scheme

Generated tables:
- Table 10: output/0307_main_comparison/table10_predicted_group_config_vs_actual_0307.csv
- Table 11: output/0307_main_comparison/table11_predicted_group_performance_vs_actual_0307.csv
- Table 12: output/0307_main_comparison/table12_per_feature_shap_difference_0307.csv
- Table 13: output/0307_main_comparison/table13_kl_js_divergence_0307.csv

Markdown previews:
- output/0307_main_comparison/table10_predicted_group_config_vs_actual_0307.md
- output/0307_main_comparison/table11_predicted_group_performance_vs_actual_0307.md
- output/0307_main_comparison/table12_per_feature_shap_difference_generated_0307.md
- output/0307_main_comparison/table13_kl_js_divergence_generated_0307.md

Scheme:
- eps2bins_3_r0307_main_tiebreak

Inputs:
- output/0307_main_comparison/hall_of_fame_0307.csv
- output/shap_eps2bins_3_r0307_main_tiebreak_niles
- output/shap_individual_manual_bins_niles
- output/new_function_task/task_H_individual_shap_affine_functions
- output/new_function_task/task_H_individual_shap_real_generated_function

Definitions:
- Predicted group configuration: the most frequent BBOB Hall-of-Fame configuration within the assigned group; ties are broken by mean AUC.
- Predicted group performance: BBOB Hall-of-Fame AUC distribution within the assigned group.
- Per-feature Wasserstein: since each feature only has one summary importance value, this equals abs(group_importance_norm - individual_importance_norm).
- KL / JS divergence: computed from normalized summary SHAP importance vectors with epsilon smoothing.

Important limitations:
- These are summary-level comparisons, not raw SHAP-value distribution comparisons.
- Configuration prediction is based on BBOB Hall-of-Fame rows only.
- Generated functions are compared against their assigned 0307 group references.
