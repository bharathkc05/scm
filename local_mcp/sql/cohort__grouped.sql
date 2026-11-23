-- Aggregate-only cohort summary over patient-level data (T1). Every statement ends in an aggregation.
-- {dims}, {metrics}, {source}, {where} and {order} are filled ONLY from the fixed whitelists in tools/computation.py;
-- no tool input ever reaches this text. Suppression and rounding are applied to the aggregated cells in Python.
SELECT {dims} count(*) AS n_admissions, count(DISTINCT subject_id) AS n_patients{metrics}
FROM {source}
WHERE {where}
GROUP BY ALL
{order}
