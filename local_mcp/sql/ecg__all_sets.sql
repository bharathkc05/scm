-- One measure across every analysis set (robustness card). $1 = measure.
SELECT * FROM l2_effects WHERE measure = $1::VARCHAR ORDER BY analysis_set
