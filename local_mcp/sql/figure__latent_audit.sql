-- Data behind the latent audit figure (top 60 by MI-explained variance). No parameters.
SELECT latent, r2_baseline, delta_r2_mi_adjusted, att_sd, top_handcrafted_feature
FROM latent_audit
ORDER BY delta_r2_mi_adjusted DESC
LIMIT 60
