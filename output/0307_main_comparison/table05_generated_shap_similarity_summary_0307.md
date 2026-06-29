# Table 5. Generated functions: SHAP similarity summary

This table summarises, for each generated function, how similar its SHAP profile is to the assigned group reference. Value distance is derived from Table 4 and shape distance is derived from Table 3.

| function_type | function_id | assigned_group_0307 | actual_auc_mean | cosine_similarity | spearman_similarity | top3_overlap | mean_value_distance | mean_shape_distance_wasserstein | mean_shape_distance_ks | most_different_hyperparameter_value | most_different_hyperparameter_shape | distribution_mode | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| affine | f10_f6 | G5 | 0.623614 | 0.884469 | 0.892941 | 1 | 0.0104066 | 0.00681686 | 0.293445 | F | F | group=raw; function=raw | ok |
| affine | f15_f8 | G1 | 0.299623 | 0.948262 | 0.874718 | 1 | 0.0115342 | 0.0096476 | 0.384114 | F | F | group=raw; function=raw | ok |
| affine | f1_f23 | G1 | 0.919491 | 0.93872 | 0.874718 | 1 | 0.0267493 | 0.0195765 | 0.313391 | F | F | group=raw; function=raw | ok |
| affine | f3_f6 | G1 | 0.429752 | 0.907733 | 0.738043 | 0.666667 | 0.011457 | 0.00621538 | 0.238155 | F | F | group=raw; function=raw | ok |
| affine | f9_f12 | G3 | 0.673026 | 0.962668 | 0.883829 | 0.666667 | 0.00911213 | 0.00421373 | 0.229914 | F | F | group=raw; function=raw | ok |
| llamea_generated | 20260516_115256 | G0 | 0.892535 | 0.89741 | 0.788157 | 0.666667 | 0.0301144 | 0.0214106 | 0.365103 | F | F | group=raw; function=raw | ok |
| llamea_generated | 20260517_125137_n3 | G1 | 0.901406 | 0.834565 | 0.66515 | 0.666667 | 0.0379414 | 0.0263125 | 0.329909 | CR | CR | group=raw; function=raw | ok |
| llamea_generated | 20260517_125137_n5 | G0 | 0.313284 | 0.918791 | 0.687929 | 0.333333 | 0.00733901 | 0.00390018 | 0.248182 | mutation_base | instance | group=raw; function=raw | ok |
| llamea_generated | 20260517_125137_n7 | G0 | 0.918892 | 0.886176 | 0.80638 | 0.666667 | 0.0330801 | 0.0241083 | 0.353132 | F | F | group=raw; function=raw | ok |
