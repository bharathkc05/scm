-- Age and sex subgroup estimates for one measure. $1 = measure.
SELECT * FROM l2_subgroups WHERE measure = $1::VARCHAR ORDER BY subgroup
