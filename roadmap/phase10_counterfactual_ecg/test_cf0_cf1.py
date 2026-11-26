"""CF0/CF1 acceptance tests: locked definitions, and internal consistency of the CF1 outputs.

    python -X utf8 -m unittest roadmap.phase10_counterfactual_ecg.test_cf0_cf1 -v
"""
import json
import unittest

import pandas as pd

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS
from roadmap.definitions import CF_PAIR_SHARES, PAIR_AFTER_GAP_DAYS, PAIR_BEFORE_GAP_HOURS, flag_codes, mi_territory

OUT = DATA_OUT / "phase10"


class Definitions(unittest.TestCase):
    def test_icd9_territories(self):
        self.assertEqual(mi_territory("41001", 9), ("anterolateral", "anterior"))
        self.assertEqual(mi_territory("41041", 9), ("inferior", "inferior"))
        self.assertEqual(mi_territory("41031", 9), ("inferoposterior", "inferior"))
        self.assertEqual(mi_territory("41071", 9), ("subendocardial", "nstemi"))
        self.assertEqual(mi_territory("410.61", 9), ("posterior", "posterior"))
        self.assertIsNone(mi_territory("41072", 9), "410.x2 is a follow-up, not a new MI")
        self.assertIsNone(mi_territory("4280", 9))

    def test_icd10_territories(self):
        self.assertEqual(mi_territory("I2102", 10), ("lad", "anterior"))
        self.assertEqual(mi_territory("I2111", 10), ("rca", "inferior"))
        self.assertEqual(mi_territory("I2121", 10), ("lcx", "lateral"))
        self.assertEqual(mi_territory("I214", 10), ("subendocardial", "nstemi"))
        self.assertEqual(mi_territory("I21A1", 10), ("type2", "type2"))
        self.assertEqual(mi_territory("I221", 10), ("inferior", "inferior"))
        self.assertIsNone(mi_territory("I252", 10), "old MI is not acute")

    def test_territory_agrees_with_phase1_acute_mi_flag(self):
        codes = pd.DataFrame({"icd_code": ["41001", "41072", "41091", "I2109", "I214", "I21A1", "I2510", "I229"],
                              "icd_version": [9, 9, 9, 10, 10, 10, 10, 10]})
        flags = flag_codes(codes)
        for (_, r), acute in zip(codes.iterrows(), flags.acute_mi):
            self.assertEqual(mi_territory(r.icd_code, r.icd_version) is not None, bool(acute), r.icd_code)

    def test_split_shares(self):
        self.assertAlmostEqual(sum(CF_PAIR_SHARES.values()), 1.0)


class CohortOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ep = pd.read_parquet(OUT / "episodes.parquet")
        cls.idx = pd.read_parquet(OUT / "ecg_index.parquet")
        cls.pairs = pd.read_parquet(OUT / "pairs.parquet")
        cls.sp = pd.read_parquet(OUT / "cf_splits.parquet")
        cls.summary = json.loads((PHASE_DIRS[10] / "cf1_01_cohort.json").read_text())

    def test_episodes_match_phase2(self):
        adm = pd.read_parquet(DATA_OUT / "phase2/admissions.parquet", columns=["hadm_id", "acute_mi"])
        self.assertEqual(set(self.ep.hadm_id), set(adm.loc[adm.acute_mi == 1, "hadm_id"]))
        self.assertTrue(self.ep.hadm_id.is_unique)
        stemi = self.ep[self.ep.mi_kind == "stemi"]
        self.assertTrue(stemi.territory.notna().all(), "every STEMI has a territory class")

    def test_ecg_index_unique_and_complete(self):
        self.assertTrue(self.idx.study_id.is_unique)
        mi_pat = set(self.ep.subject_id)
        controls = self.idx[self.idx.role == "control"]
        self.assertFalse(set(controls.subject_id) & mi_pat, "controls never had an MI admission")
        self.assertTrue(controls.subject_id.is_unique)
        self.assertTrue((controls.age_at_ecg >= 18).all())

    def test_pair_timing_rules(self):
        p = self.pairs
        self.assertTrue((p.before_study_id.notna() | p.after_study_id.notna()).all())
        b = p.dropna(subset=["before_ecg_time"])
        self.assertTrue((b.before_ecg_time < b.during_ecg_time).all())
        ep = self.ep.set_index("hadm_id")
        arrival = ep.loc[b.hadm_id, "arrival_time"].to_numpy()
        self.assertTrue(((arrival - b.before_ecg_time.to_numpy()) >= pd.Timedelta(hours=PAIR_BEFORE_GAP_HOURS)).all())
        a = p.dropna(subset=["after_ecg_time"])
        disch = ep.loc[a.hadm_id, "dischtime"].to_numpy()
        self.assertTrue(((a.after_ecg_time.to_numpy() - disch) > pd.Timedelta(days=PAIR_AFTER_GAP_DAYS)).all())
        during = self.idx.set_index("study_id").loc[p.during_study_id]
        self.assertTrue(during.role.str.contains("during").all())
        self.assertTrue(during.during_hadm_id.notna().all())

    def test_baselines_are_outside_every_mi_stay(self):
        base = self.idx[self.idx.role.str.contains("baseline")]
        self.assertTrue(base.during_hadm_id.isna().all())

    def test_splits_patient_level_and_frozen(self):
        self.assertTrue(self.sp.subject_id.is_unique)
        pair_pat = set(self.pairs.subject_id)
        held = self.sp[self.sp.cf_split.isin(["pair_test", "pair_val"])]
        self.assertEqual(set(held.subject_id), pair_pat)
        share = (held.cf_split == "pair_test").mean()
        self.assertAlmostEqual(share, CF_PAIR_SHARES["pair_test"], delta=0.01)
        train = set(self.sp.loc[self.sp.cf_split == "train", "subject_id"])
        self.assertFalse(train & pair_pat, "no pair patient in the Phase 10 training split")
        self.assertTrue(self.summary["gate_cf_g0"]["pass"])

    def test_no_identifiers_in_committed_summary(self):
        from local_mcp import egress
        from roadmap.config import SEED
        # the fixed random seed (20261121) has the shape of a MIMIC identifier; it is a constant, not a patient
        text = (PHASE_DIRS[10] / "cf1_01_cohort.json").read_text().replace(str(SEED), "<seed>")
        self.assertEqual(egress.scan_text(text), [])


if __name__ == "__main__":
    unittest.main()
