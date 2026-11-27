-- Data behind the pair validation figure: per-measure absolute error averaged over the 14 ST/T measures,
-- primary held-out pairs. No parameters.
SELECT direction, method, avg(mae_sd) AS mean_abs_error_sd, count(*) AS measures, max(n) AS pairs
FROM cf_pair_validation
WHERE "set" = 'primary' AND "group" IN ('ST', 'T')
GROUP BY direction, method
ORDER BY direction, method
