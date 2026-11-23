-- One Layer 2 measure in one analysis set, with its Phase 6 classification. $1 = analysis_set, $2 = measure.
SELECT e.*, c.class, c.robust_mi_sign, c.adjusted_share_of_crude, c.nstemi_same_sign, c.negative_control_ratio
FROM l2_effects e
LEFT JOIN l2_class c USING (measure)
WHERE e.analysis_set = $1::VARCHAR AND e.measure = $2::VARCHAR
