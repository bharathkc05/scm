-- Phase 7 E-value for one Layer 2 measure. $1 = measure.
SELECT evalue_point, evalue_ci, approx_rr FROM evalues WHERE layer = 2 AND outcome = $1::VARCHAR
