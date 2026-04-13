Group vs Individual SHAP Consistency Summary
============================================================

Top-k used for overlap: 5

Per-group consistency:
 group_id         group_label  n_features  n_functions_in_group  spearman_rank_corr  top_5_overlap_count  top_5_jaccard                                             group_top_features                                               individual_mean_top_features
        0 eps_bin_0__r2_bin_0          16                     7            0.970588                    4       0.666667                   CR | F | lambda_ | lpsr | mutation_base_best                                 CR | F | lambda_ | lpsr | mutation_n_comps
        1 eps_bin_0__r2_bin_1          16                     7            0.908824                    4       0.666667     F | lambda_ | lpsr | mutation_base_best | mutation_n_comps                               CR | F | lambda_ | lpsr | mutation_base_best
        2 eps_bin_0__r2_bin_2          16                     2            0.876471                    4       0.666667 F | lambda_ | lpsr | mutation_base_best | mutation_base_target F | lambda_ | mutation_base_best | mutation_base_target | mutation_n_comps
        3 eps_bin_1__r2_bin_0          16                     3            0.970588                    4       0.666667     F | lambda_ | lpsr | mutation_base_best | mutation_n_comps                               CR | F | lambda_ | lpsr | mutation_base_best
        4 eps_bin_1__r2_bin_1          16                     3            0.979412                    4       0.666667                   CR | F | lambda_ | lpsr | mutation_base_best                                 CR | F | lambda_ | lpsr | mutation_n_comps
        5 eps_bin_1__r2_bin_2          16                     2            0.967647                    5       1.000000                   CR | F | lambda_ | lpsr | mutation_base_best                               CR | F | lambda_ | lpsr | mutation_base_best

Global individual-level mean importance:
                 feature  global_individual_mean_importance  global_individual_std_importance  n_functions
                       F                           0.035280                          0.026969           24
                 lambda_                           0.019055                          0.017674           24
                      CR                           0.013658                          0.009714           24
                    lpsr                           0.013521                          0.012815           24
      mutation_base_best                           0.010467                          0.010249           24
        mutation_n_comps                           0.009802                          0.006702           24
    mutation_base_target                           0.007283                          0.016196           24
      mutation_base_rand                           0.003637                          0.003951           24
             use_archive                           0.003609                          0.004584           24
  mutation_reference_nan                           0.002407                          0.001685           24
 mutation_reference_best                           0.001251                          0.001165           24
 mutation_reference_rand                           0.001109                          0.001918           24
           crossover_exp                           0.000975                          0.000923           24
           crossover_bin                           0.000938                          0.000876           24
mutation_reference_pbest                           0.000908                          0.000661           24
   adaptation_method_nan                           0.000000                          0.000000           24