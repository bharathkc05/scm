-- Latent audit row for one dimension. $1 = latent name (z_ecg_k).
SELECT latent, r2_baseline, r2_mi_alone, delta_r2_mi_adjusted, att_sd, att_q, top_handcrafted_feature, top_feature_corr
FROM latent_audit
WHERE latent = $1::VARCHAR
