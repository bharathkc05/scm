-- Layer 1 derived views: one row per (result, estimand), and every Phase 7 check per result.
CREATE VIEW l1_effects AS
SELECT l1_result_id(label)                          AS result_id,
       split_part(l1_result_id(label), ':', 2)      AS exposure,
       split_part(l1_result_id(label), ':', 3)      AS adjustment,
       label, role, lower(estimand)                 AS estimand, adjust_for, note,
       n, exposed, crude_rd_pp, rd_pp, rd_ci_low_pp, rd_ci_high_pp, rr, rr_ci_low, rr_ci_high
FROM layer1_estimates;

CREATE VIEW l1_checks AS
SELECT l1_result_id(label) AS result_id, 'outcome_codes' AS check_group, variant AS check_name,
       n, exposed, outcome_events, rd_pp, rr, rr_ci_low, rr_ci_high
FROM outcome_codes_layer1
UNION ALL
SELECT l1_result_id(label), 'subgroup', subgroup, n, exposed, outcome_events, rd_pp, rr, rr_ci_low, rr_ci_high
FROM subgroups_layer1
UNION ALL
SELECT l1_result_id(label), 'history_definition', variant, n, exposed, outcome_events, rd_pp, rr, rr_ci_low, rr_ci_high
FROM layer1_history_variants;
