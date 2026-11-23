-- Layer 2 derived views: every ATT estimate under one schema, keyed by (analysis_set, measure).
CREATE VIEW l2_effects AS
SELECT 'primary' AS analysis_set, measure_name(outcome) AS measure, family, "group" AS grp, control_sd, crude_sd,
       att, att_ci_low, att_ci_high, att_sd, att_sd_ci_low, att_sd_ci_high, att_p, att_q
FROM layer2_estimates
WHERE family <> 'raw_point'
UNION ALL
SELECT l2_set_name(analysis_set), measure_name(outcome), family, "group", control_sd, crude_sd,
       att, att_ci_low, att_ci_high, att_sd, att_sd_ci_low, att_sd_ci_high, att_p, att_q
FROM sensitivity_estimates
UNION ALL
SELECT 'selection_weighted', s.outcome, p.family, p."group", p.control_sd, p.crude_sd,
       s.att_sd * p.control_sd, s.att_sd_ci_low * p.control_sd, s.att_sd_ci_high * p.control_sd,
       s.att_sd, s.att_sd_ci_low, s.att_sd_ci_high, s.att_p, NULL
FROM selection_weighted_layer2 s
JOIN layer2_estimates p ON measure_name(p.outcome) = s.outcome AND p.family IN ('point', 'st_composite')
UNION ALL
SELECT 'bmi_adjusted', measure_name(outcome), family, "group", control_sd, crude_sd,
       att, att_ci_low, att_ci_high, att_sd, att_sd_ci_low, att_sd_ci_high, att_p, NULL
FROM bmi_layer2_with
UNION ALL
SELECT 'bmi_subset_unadjusted', measure_name(outcome), family, "group", control_sd, crude_sd,
       att, att_ci_low, att_ci_high, att_sd, att_sd_ci_low, att_sd_ci_high, att_p, NULL
FROM bmi_layer2_without
UNION ALL
SELECT 'broad_codes', measure_name(outcome), family, "group", control_sd, crude_sd,
       att, att_ci_low, att_ci_high, att_sd, att_sd_ci_low, att_sd_ci_high, att_p, NULL
FROM outcome_codes_layer2_broad;

CREATE VIEW l2_class AS
SELECT measure_name(outcome) AS measure, class, robust_mi_sign, adjusted_share_of_crude,
       nstemi_same_sign, negative_control_ratio, crude_q
FROM deconfounded_ecg_effects;

CREATE VIEW l2_subgroups AS
SELECT subgroup, measure_name(outcome) AS measure, family, "group" AS grp,
       att, att_ci_low, att_ci_high, att_sd, att_sd_ci_low, att_sd_ci_high, att_p
FROM subgroups_layer2;
