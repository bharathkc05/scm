-- Manuscript claim map (Phase 8). No parameters.
SELECT claim_id, manuscript_section, claim, evidence_files, assumptions, strength
FROM claim_evidence_map
ORDER BY claim_id
