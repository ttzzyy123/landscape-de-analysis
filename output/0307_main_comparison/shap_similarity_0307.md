# SHAP similarity under 0307 main tiebreak scheme

This analysis compares function-level SHAP importance profiles against the assigned group-level SHAP reference.

Scheme: `eps2bins_3_r0307_main_tiebreak`

Important: ambiguous-on-tie and mean-based variants are not used.

BBOB individual SHAP source: `output/shap_individual_manual_bins_niles/`.

Wasserstein note: current files contain summary-level importance values rather than raw SHAP-value distributions. The Wasserstein columns are therefore computed from normalized summary importance vectors.

## Function-type summary

| function_type | n_functions_ok | mean_cosine_similarity | median_cosine_similarity | mean_spearman_rank_corr | median_spearman_rank_corr | mean_top3_overlap_ratio | median_top3_overlap_ratio | mean_l1_distance | median_l1_distance | mean_wasserstein | median_wasserstein | mean_wasserstein_importance_distribution | median_wasserstein_importance_distribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bbob_original | 24 | 0.953432 | 0.968216 | 0.857576 | 0.890909 | 0.833333 | 1 | 0.287462 | 0.270249 | 0.0261329 | 0.0185909 | 0.0229106 | 0.021981 |
| affine | 5 | 0.948319 | 0.955212 | 0.847273 | 0.881818 | 0.866667 | 1 | 0.295319 | 0.341332 | 0.0268472 | 0.0177567 | 0.0241477 | 0.0229496 |
| llamea_generated | 4 | 0.912824 | 0.911003 | 0.729545 | 0.740909 | 0.5 | 0.5 | 0.395434 | 0.383192 | 0.0359486 | 0.0265765 | 0.0298541 | 0.0294927 |

## Group summary

| assigned_group_0307 | n_functions_ok | function_types | mean_cosine_similarity | median_cosine_similarity | mean_spearman_rank_corr | median_spearman_rank_corr | mean_top3_overlap_ratio | median_top3_overlap_ratio | mean_l1_distance | median_l1_distance | mean_wasserstein | median_wasserstein | mean_wasserstein_importance_distribution | median_wasserstein_importance_distribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | 9 | bbob_original,llamea_generated | 0.950337 | 0.949907 | 0.79596 | 0.790909 | 0.740741 | 0.666667 | 0.306315 | 0.305356 | 0.0278468 | 0.018484 | 0.0259449 | 0.0245167 |
| G1 | 12 | affine,bbob_original,llamea_generated | 0.949032 | 0.96 | 0.843939 | 0.868182 | 0.833333 | 0.833333 | 0.302263 | 0.307988 | 0.0274784 | 0.0192584 | 0.0236579 | 0.0225876 |
| G2 | 2 | bbob_original | 0.890795 | 0.890795 | 0.677273 | 0.677273 | 0.833333 | 0.833333 | 0.543759 | 0.543759 | 0.0494326 | 0.0415446 | 0.0416897 | 0.0416897 |
| G3 | 3 | affine,bbob_original | 0.982151 | 0.983891 | 0.915152 | 0.936364 | 0.777778 | 0.666667 | 0.194289 | 0.176944 | 0.0176626 | 0.0121988 | 0.0143505 | 0.0155851 |
| G4 | 4 | bbob_original | 0.949318 | 0.981659 | 0.909091 | 0.927273 | 0.833333 | 1 | 0.244258 | 0.207614 | 0.0222052 | 0.0145069 | 0.0175123 | 0.0174423 |
| G5 | 3 | affine,bbob_original | 0.936176 | 0.923424 | 0.90303 | 0.890909 | 0.777778 | 0.666667 | 0.308672 | 0.324621 | 0.0280611 | 0.0200337 | 0.0253772 | 0.0264607 |

## Per-function similarity

| function_type | function_id | assigned_group_0307 | actual_auc_mean | cosine_similarity | spearman_rank_corr | l1_distance | l2_distance | top3_overlap_ratio | wasserstein_importance_distribution | mean_wasserstein | max_wasserstein | individual_top3_features | group_top3_features | most_different_feature | most_different_abs_diff | most_different_feature_wasserstein | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bbob_original | f1 | G2 | 0.938472 | 0.91407 | 0.827273 | 0.496827 | 0.167495 | 1 | 0.0362767 | 0.0451661 | 0.102501 | F,lambda_,mutation_base | mutation_base,F,lambda_ | F | 0.102501 | F | ok |
| bbob_original | f10 | G5 | 0.703066 | 0.923424 | 0.890909 | 0.324621 | 0.151792 | 0.666667 | 0.0264607 | 0.029511 | 0.134369 | CR,F,lpsr | F,CR,lambda_ | CR | 0.134369 | CR | ok |
| bbob_original | f11 | G4 | 0.759576 | 0.845135 | 0.809091 | 0.416821 | 0.20603 | 0.333333 | 0.0219845 | 0.0378929 | 0.174769 | CR,F,mutation_base | F,lambda_,lpsr | CR | 0.174769 | CR | ok |
| bbob_original | f12 | G3 | 0.370939 | 0.983891 | 0.936364 | 0.175239 | 0.0668751 | 0.666667 | 0.0115834 | 0.0159308 | 0.0482758 | F,mutation_base,CR | F,mutation_base,lambda_ | CR | 0.0482758 | CR | ok |
| bbob_original | f13 | G1 | 0.652542 | 0.964852 | 0.854545 | 0.273431 | 0.102726 | 0.666667 | 0.0248574 | 0.0248574 | 0.0637628 | F,lambda_,CR | F,lambda_,mutation_base | CR | 0.0637628 | CR | ok |
| bbob_original | f14 | G1 | 0.870898 | 0.964787 | 0.936364 | 0.329962 | 0.113619 | 1 | 0.0299965 | 0.0299965 | 0.0663836 | F,lambda_,mutation_base | F,lambda_,mutation_base | F | 0.0663836 | F | ok |
| bbob_original | f15 | G1 | 0.177028 | 0.974159 | 0.936364 | 0.239763 | 0.093369 | 1 | 0.0198812 | 0.0217966 | 0.0692004 | F,lambda_,mutation_base | F,lambda_,mutation_base | F | 0.0692004 | F | ok |
| bbob_original | f16 | G0 | 0.431921 | 0.958772 | 0.790909 | 0.295223 | 0.11122 | 0.666667 | 0.0245167 | 0.0268384 | 0.0698773 | F,mutation_base,CR | F,mutation_base,mutation_n_comps | mutation_base | 0.0698773 | mutation_base | ok |
| bbob_original | f17 | G1 | 0.677192 | 0.974874 | 0.927273 | 0.256438 | 0.0897584 | 1 | 0.0213706 | 0.0233126 | 0.055058 | F,mutation_base,lambda_ | F,lambda_,mutation_base | F | 0.055058 | F | ok |
| bbob_original | f18 | G1 | 0.604785 | 0.972742 | 0.927273 | 0.267066 | 0.0934342 | 1 | 0.0222257 | 0.0242787 | 0.0605705 | F,mutation_base,lambda_ | F,lambda_,mutation_base | F | 0.0605705 | F | ok |
| bbob_original | f19 | G0 | 0.241373 | 0.939628 | 0.8 | 0.323244 | 0.145635 | 1 | 0.0277952 | 0.0293858 | 0.125496 | F,mutation_n_comps,mutation_base | F,mutation_base,mutation_n_comps | F | 0.125496 | F | ok |
| bbob_original | f2 | G4 | 0.82998 | 0.984873 | 0.972727 | 0.220196 | 0.0763501 | 1 | 0.0200178 | 0.0200178 | 0.0363439 | F,lambda_,lpsr | F,lambda_,lpsr | lpsr | 0.0363439 | lpsr | ok |
| bbob_original | f20 | G4 | 0.451167 | 0.978445 | 0.890909 | 0.195031 | 0.0741536 | 1 | 0.0148667 | 0.0177301 | 0.0493824 | F,lambda_,lpsr | F,lambda_,lpsr | CR | 0.0493824 | CR | ok |
| bbob_original | f21 | G0 | 0.707107 | 0.977526 | 0.954545 | 0.213107 | 0.0799132 | 1 | 0.0186056 | 0.0193733 | 0.0574831 | F,mutation_base,mutation_n_comps | F,mutation_base,mutation_n_comps | F | 0.0574831 | F | ok |
| bbob_original | f22 | G0 | 0.656389 | 0.988278 | 0.954545 | 0.15782 | 0.0554512 | 1 | 0.0127862 | 0.0143472 | 0.030175 | F,mutation_base,mutation_n_comps | F,mutation_base,mutation_n_comps | F | 0.030175 | F | ok |
| bbob_original | f23 | G0 | 0.251983 | 0.976552 | 0.618182 | 0.191731 | 0.0710144 | 0.666667 | 0.0131351 | 0.0174301 | 0.053759 | F,mutation_base,Instance variance | F,mutation_base,mutation_n_comps | F | 0.053759 | F | ok |
| bbob_original | f24 | G0 | 0.104046 | 0.940367 | 0.845455 | 0.350874 | 0.145077 | 1 | 0.0318976 | 0.0318976 | 0.116561 | F,mutation_n_comps,mutation_base | F,mutation_base,mutation_n_comps | F | 0.116561 | F | ok |
| bbob_original | f3 | G1 | 0.691577 | 0.926588 | 0.763636 | 0.360298 | 0.133279 | 0.666667 | 0.0291349 | 0.0327543 | 0.0894764 | F,CR,lambda_ | F,lambda_,mutation_base | CR | 0.0894764 | CR | ok |
| bbob_original | f4 | G1 | 0.610472 | 0.925876 | 0.709091 | 0.360364 | 0.135177 | 0.666667 | 0.0266732 | 0.0327604 | 0.0964019 | F,CR,lambda_ | F,lambda_,mutation_base | CR | 0.0964019 | CR | ok |
| bbob_original | f5 | G2 | 0.991066 | 0.867521 | 0.527273 | 0.59069 | 0.233906 | 0.666667 | 0.0471028 | 0.0536991 | 0.179661 | F,mutation_base,use_archive | mutation_base,F,lambda_ | F | 0.179661 | F | ok |
| bbob_original | f6 | G5 | 0.778969 | 0.97158 | 0.936364 | 0.252402 | 0.090327 | 0.666667 | 0.017944 | 0.0229456 | 0.0540376 | F,lambda_,lpsr | F,CR,lambda_ | F | 0.0540376 | F | ok |
| bbob_original | f7 | G1 | 0.897539 | 0.950938 | 0.845455 | 0.286014 | 0.119258 | 0.666667 | 0.0219775 | 0.0260012 | 0.0877963 | F,CR,lambda_ | F,lambda_,mutation_base | CR | 0.0877963 | CR | ok |
| bbob_original | f8 | G4 | 0.598859 | 0.98882 | 0.963636 | 0.144981 | 0.0610143 | 1 | 0.0131801 | 0.0131801 | 0.0496418 | F,lambda_,lpsr | F,lambda_,lpsr | F | 0.0496418 | F | ok |
| bbob_original | f9 | G3 | 0.62686 | 0.988671 | 0.963636 | 0.176944 | 0.0662164 | 1 | 0.0155851 | 0.0160858 | 0.0484308 | F,lambda_,mutation_base | F,mutation_base,lambda_ | F | 0.0484308 | F | ok |
| affine | f10_f6 | G5 | 0.623614 | 0.913524 | 0.881818 | 0.348994 | 0.165963 | 1 | 0.0317267 | 0.0317267 | 0.144763 | CR,F,lambda_ | F,CR,lambda_ | CR | 0.144763 | CR | ok |
| affine | f15_f8 | G1 | 0.299623 | 0.975943 | 0.909091 | 0.194562 | 0.0833287 | 1 | 0.017359 | 0.0176875 | 0.0635451 | F,lambda_,mutation_base | F,lambda_,mutation_base | Instance variance | 0.0635451 | Instance variance | ok |
| affine | f1_f23 | G1 | 0.919491 | 0.955212 | 0.881818 | 0.361022 | 0.126035 | 1 | 0.0328202 | 0.0328202 | 0.0635451 | F,lambda_,mutation_base | F,lambda_,mutation_base | Instance variance | 0.0635451 | Instance variance | ok |
| affine | f3_f6 | G1 | 0.429752 | 0.923028 | 0.718182 | 0.341332 | 0.135487 | 0.666667 | 0.0229496 | 0.0310302 | 0.0795596 | lambda_,CR,F | F,lambda_,mutation_base | CR | 0.0795596 | CR | ok |
| affine | f9_f12 | G3 | 0.673026 | 0.97389 | 0.845455 | 0.230683 | 0.0856365 | 0.666667 | 0.0158831 | 0.0209712 | 0.0482758 | F,lambda_,CR | F,mutation_base,lambda_ | Instance variance | 0.0482758 | Instance variance | ok |
| llamea_generated | 20260516_115256 | G0 | 0.892535 | 0.919349 | 0.763636 | 0.409484 | 0.152368 | 0.333333 | 0.0358329 | 0.0372258 | 0.0808173 | F,lambda_,lpsr | F,mutation_base,mutation_n_comps | lambda_ | 0.0808173 | lambda_ | ok |
| llamea_generated | 20260517_125137_n3 | G1 | 0.901406 | 0.879383 | 0.718182 | 0.356899 | 0.171771 | 0.666667 | 0.0146489 | 0.0324454 | 0.144406 | CR,F,lambda_ | F,lambda_,mutation_base | CR | 0.144406 | CR | ok |
| llamea_generated | 20260517_125137_n5 | G0 | 0.313284 | 0.949907 | 0.645455 | 0.305356 | 0.116769 | 0.333333 | 0.0231526 | 0.0277596 | 0.0795381 | F,CR,lambda_ | F,mutation_base,mutation_n_comps | Instance variance | 0.0795381 | Instance variance | ok |
| llamea_generated | 20260517_125137_n7 | G0 | 0.918892 | 0.902658 | 0.790909 | 0.509997 | 0.181144 | 0.666667 | 0.0457822 | 0.0463634 | 0.100625 | F,lambda_,mutation_base | F,mutation_base,mutation_n_comps | lambda_ | 0.100625 | lambda_ | ok |

## Error rows

No error rows.
