-- Phase 10 Track A pair validation, one row per measure and method. $1 = pair set (primary | all_before | all_after),
-- $2 = direction (add | remove).
SELECT * FROM cf_pair_validation WHERE "set" = $1::VARCHAR AND direction = $2::VARCHAR ORDER BY measure, method
