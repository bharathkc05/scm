-- One Layer 1 estimate. $1 = result_id, $2 = estimand (ate | ate_overlap | att).
SELECT * FROM l1_effects WHERE result_id = $1::VARCHAR AND estimand = $2::VARCHAR
