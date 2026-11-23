-- Latent audit summary. No parameters.
SELECT count(*) AS latents,
       count(*) FILTER (WHERE att_q < 0.05) AS fdr_significant,
       median(r2_baseline) AS median_r2_baseline,
       median(delta_r2_mi_adjusted) AS median_delta_r2_mi,
       max(delta_r2_mi_adjusted) AS max_delta_r2_mi,
       count(*) FILTER (WHERE delta_r2_mi_adjusted > r2_baseline) AS mi_exceeds_baseline
FROM latent_audit
