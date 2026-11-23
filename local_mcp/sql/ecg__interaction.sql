-- Phase 7 interaction test (16 robust measures only). $1 = measure, $2 = modifier (age | sex).
SELECT * FROM subgroup_interactions_layer2 WHERE outcome = $1::VARCHAR AND modifier = $2::VARCHAR
