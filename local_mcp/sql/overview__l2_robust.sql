-- Robust MI signs (Phase 6 classification), largest first. No parameters.
SELECT e.measure, e.family, e.att_sd, e.att_sd_ci_low, e.att_sd_ci_high
FROM l2_effects e
JOIN l2_class c USING (measure)
WHERE e.analysis_set = 'primary' AND c.robust_mi_sign
ORDER BY abs(e.att_sd) DESC
