-- Summary of a family/group. $1 = analysis_set, $2 = family, $3 = group (NULL = all).
SELECT count(*) AS measures,
       count(*) FILTER (WHERE att_q < 0.05) AS fdr_significant,
       count(*) FILTER (WHERE att_sd_ci_low > 0 OR att_sd_ci_high < 0) AS ci_excludes_0,
       avg(abs(att_sd)) AS mean_abs_att_sd
FROM l2_effects
WHERE analysis_set = $1::VARCHAR AND family = $2::VARCHAR AND ($3::VARCHAR IS NULL OR grp = $3::VARCHAR)
