-- Phase 7 checks for one Layer 1 result. $1 = result_id.
SELECT * FROM l1_checks WHERE result_id = $1::VARCHAR ORDER BY check_group, check_name
