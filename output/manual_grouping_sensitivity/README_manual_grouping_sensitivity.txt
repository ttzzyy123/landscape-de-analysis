Manual bin grouping sensitivity experiment summary
============================================================

Feature file: intermediate/dim5_selected_features.csv
Features used: eps_ratio, adj_r2

Overall grouping summary:
     method          config                       eps_edges                    adj_r2_edges  n_eps_bins  n_r2_bins  n_functions  n_groups  min_group_size  max_group_size  mean_group_size  std_group_size  singleton_groups  small_groups_le_2  small_groups_le_3  grouping_flag
manual_bins manual_2x3_main      [-Infinity, 3.0, Infinity] [-Infinity, 0.3, 0.7, Infinity]           2          3           24         6               2               7              4.0        2.160247                 0                  2                  4     reasonable
manual_bins manual_2x3_alt1      [-Infinity, 3.0, Infinity] [-Infinity, 0.2, 0.6, Infinity]           2          3           24         6               1               6              4.0        1.914854                 1                  2                  2 has_singletons
manual_bins manual_3x3_alt2 [-Infinity, 2.5, 5.0, Infinity] [-Infinity, 0.2, 0.6, Infinity]           3          3           24         8               1               6              3.0        2.000000                 3                  4                  5 has_singletons
manual_bins manual_2x2_alt3      [-Infinity, 3.0, Infinity]      [-Infinity, 0.5, Infinity]           2          2           24         4               3              11              6.0        3.000000                 0                  0                  1     reasonable

Top pairwise ARI (higher = more similar grouping):
       config_a        config_b  n_common_functions      ARI
manual_2x3_alt1 manual_3x3_alt2                  24 0.904602
manual_2x3_alt1 manual_2x2_alt3                  24 0.583481
manual_2x3_main manual_3x3_alt2                  24 0.549712
manual_2x3_main manual_2x3_alt1                  24 0.512323
manual_3x3_alt2 manual_2x2_alt3                  24 0.496809
manual_2x3_main manual_2x2_alt3                  24 0.398431
