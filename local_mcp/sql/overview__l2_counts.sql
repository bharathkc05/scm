-- Layer 2 primary family sizes and FDR-significant counts. No parameters.
SELECT family, count(*) AS tests, count(*) FILTER (WHERE att_q < 0.05) AS significant
FROM l2_effects
WHERE analysis_set = 'primary' AND family IN ('point', 'iqr', 'latent')
GROUP BY family
ORDER BY family
