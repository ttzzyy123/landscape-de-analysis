Step 9 Hall of Fame under the 0307 main grouping scheme

Purpose:
- Reproduce the Hall-of-Fame idea from Table 8 of the explainable benchmarking paper.
- Use only existing results; this script does not rerun MODDE or SHAP.
- Align all outputs to eps2bins_3_r0307_main_tiebreak.

Outputs:
- output/0307_main_comparison/hall_of_fame_0307.csv
- output/0307_main_comparison/hall_of_fame_0307.md

Selection rule:
- For each function, group rows by DE configuration.
- Average auc over repeated seeds / rows.
- Pick the configuration with the highest mean auc.

Row counts by function_type:
{
  "bbob_original": 25,
  "affine": 6,
  "llamea_generated": 5
}

Note:
- LLaMEA generated functions may have older assignment files from eps2bins_3_r025065.
- For 0307 alignment, this script recomputes assigned_group_0307 from eps_ratio and adj_r2.