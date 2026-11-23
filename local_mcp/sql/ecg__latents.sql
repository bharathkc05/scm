-- Latent audit, dimensions where MI explains the most variance. $1 = top_k.
SELECT latent, r2_baseline, r2_mi_alone, delta_r2_mi_adjusted, att_sd, att_q, top_handcrafted_feature, top_feature_corr
FROM latent_audit
ORDER BY delta_r2_mi_adjusted DESC
LIMIT $1::INTEGER
