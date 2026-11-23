-- Data behind the counterfactual profiles figure. No parameters.
SELECT profile, expected_without_mi, expected_with_mi, model_implied_mi_effect, ci_low, ci_high, effect_sd
FROM counterfactual_profiles
WHERE outcome = 'ST_lateral_mean'
