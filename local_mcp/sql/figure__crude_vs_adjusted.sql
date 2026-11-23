-- Data behind the crude vs adjusted figure. No parameters.
SELECT outcome AS measure, "group", crude_sd, att_sd, class
FROM deconfounded_ecg_effects
WHERE family = 'point'
ORDER BY abs(att_sd) DESC
