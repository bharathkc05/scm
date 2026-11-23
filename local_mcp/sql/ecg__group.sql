-- Largest effects in a family/group. $1 = analysis_set, $2 = family, $3 = group (NULL = all), $4 = top_k.
SELECT e.*, c.class, c.robust_mi_sign, c.adjusted_share_of_crude, c.nstemi_same_sign, c.negative_control_ratio
FROM l2_effects e
LEFT JOIN l2_class c USING (measure)
WHERE e.analysis_set = $1::VARCHAR AND e.family = $2::VARCHAR AND ($3::VARCHAR IS NULL OR e.grp = $3::VARCHAR)
ORDER BY abs(e.att_sd) DESC
LIMIT $4::INTEGER
