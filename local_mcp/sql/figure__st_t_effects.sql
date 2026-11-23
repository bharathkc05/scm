-- Data behind the ST/T effects figure. No parameters.
SELECT outcome AS measure, "group", att_sd, att_sd_ci_low, att_sd_ci_high, nstemi_att_sd, negative_control_att_sd
FROM deconfounded_ecg_effects
WHERE family = 'point' AND "group" IN ('ST', 'T')
ORDER BY "group", att_sd
