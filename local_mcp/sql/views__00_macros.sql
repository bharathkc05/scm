-- Shared macros for the derived views. Run once at sandbox start (no parameters).
CREATE MACRO measure_name(o) AS regexp_replace(o, '^w_', '');

CREATE MACRO l1_result_id(lbl) AS CASE lbl
    WHEN 'diabetes | age, sex'               THEN 'L1:diabetes:primary'
    WHEN 'diabetes | age, sex, HTN'          THEN 'L1:diabetes:alternative'
    WHEN 'hypertension | age, sex, diabetes' THEN 'L1:hypertension:primary'
    WHEN 'hypertension | age, sex (roadmap)' THEN 'L1:hypertension:alternative'
    WHEN 'CKD | age, sex, D, H'              THEN 'L1:ckd:primary'
    WHEN 'CKD | age, sex, D, H, prior MI'    THEN 'L1:ckd:alternative'
    WHEN 'prior MI | age, sex, D, H, CKD'    THEN 'L1:prior_mi:primary'
END;

CREATE MACRO l2_set_name(s) AS CASE s
    WHEN 'nstemi_only'                  THEN 'nstemi'
    WHEN 'active_comparator_chest_pain' THEN 'chest_pain_comparator'
    WHEN 'after_strict_timing'          THEN 'after_strict'
    WHEN 'type2_excluded'               THEN 'no_type2'
    WHEN 'first_ecg_has_features'       THEN 'first_ecg_has_features'
    WHEN 'negative_control_future_mi'   THEN 'negative_control'
END;
