"""CF3 acceptance tests: Track A training/evaluation separation, model identities, and committed outputs.

    python -X utf8 -m unittest roadmap.phase10_counterfactual_ecg.test_cf3 -v
"""
import json
import pickle
import unittest

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS
from roadmap.phase10_counterfactual_ecg import cf3_01_track_a as cf3

OUT = DATA_OUT / "phase10"
DOC = PHASE_DIRS[10]


class TrackA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meas = cf3.measures()
        cls.ep = pd.read_parquet(OUT / "episodes.parquet")
        cls.idx = pd.read_parquet(OUT / "ecg_index.parquet")
        cls.pairs = pd.read_parquet(OUT / "pairs.parquet")
        cls.split = pd.read_parquet(OUT / "cf_splits.parquet").set_index("subject_id").cf_split.astype(object)
        cls.tr = cf3.training_rows(cls.meas, cls.ep, cls.idx, cls.split)
        cls.p = cf3.pair_table(cls.meas, cls.ep, cls.pairs, cls.split)
        with open(OUT / "cf3_track_a_model.pkl", "rb") as fh:
            cls.saved = pickle.load(fh)
        cls.summary = json.loads((DOC / "cf3_01_track_a.json").read_text())

    def test_training_uses_only_training_split(self):
        self.assertTrue((self.tr.subject_id.map(self.split) == "train").all())
        self.assertFalse(set(self.tr.subject_id) & set(self.pairs.subject_id))
        mi = self.tr[self.tr.mi == 1]
        self.assertTrue(mi.study_id.is_unique)
        self.assertTrue(set(mi.cf_type) <= set(cf3.TYPES))

    def test_measures_cleaned_and_winsorized(self):
        bounds = json.loads((PHASE_DIRS[3] / "winsorization_bounds.json").read_text())["bounds"]
        for m in cf3.ROBUST:
            v = self.meas[m].dropna()
            self.assertGreaterEqual(v.min(), bounds[m][0] - 1e-9, m)
            self.assertLessEqual(v.max(), bounds[m][1] + 1e-9, m)

    def test_composites(self):
        f = pd.read_parquet(OUT / "ecgdeli_features.parquet",
                            columns=["record_id", "dataset"] + [f"ST_Elev_{l}" for l in cf3.LEADS])
        f = f[f.dataset == "mimic"].drop(columns="dataset").set_index("record_id").loc[self.meas.index[:500]]
        st = f.where(f.abs() <= cf3.ST_MAX_ABS_MV)
        lat = st[[f"ST_Elev_{l}" for l in cf3.LATERAL]].mean(axis=1)
        bounds = json.loads((PHASE_DIRS[3] / "winsorization_bounds.json").read_text())["bounds"]
        np.testing.assert_allclose(lat.clip(*bounds["ST_lateral_mean"]), self.meas.ST_lateral_mean.iloc[:500])

    def test_pairs_split_and_primary_definition(self):
        self.assertTrue(self.p.cf_split.isin(["pair_test", "pair_val"]).all())
        prim = self.p[self.p.primary]
        self.assertTrue((prim.side == "before").all())
        self.assertEqual(int((prim.cf_split == "pair_test").sum()), self.summary["pairs"]["primary_test_pairs"])
        self.assertTrue(self.p.during_study_id.isin(self.meas.index).all())

    def test_additive_counterfactual_identity(self):
        """A1 with rho = 1 equals the observed value minus/plus tau, so add and remove errors are mirror images."""
        model = self.saved["model"]
        sub = self.p[self.p.primary].head(200).reset_index(drop=True)
        d = cf3.predict(model, sub, self.meas)
        tau = model.m1(sub) - model.m0(sub)
        a1_add = cf3.methods(d["add"], np.ones(16))["A1_additive"]
        a1_rem = cf3.methods(d["remove"], np.ones(16))["A1_additive"]
        np.testing.assert_allclose(a1_add, d["add"]["src"] + tau)
        np.testing.assert_allclose(a1_rem, d["remove"]["src"] - tau)
        np.testing.assert_allclose(np.abs(a1_add - d["add"]["tgt"]), np.abs(a1_rem - d["remove"]["tgt"]))

    def test_rho_and_calibration_come_from_pair_val(self):
        n_val = self.p[self.p.cf_split == "pair_val"].groupby("side").size()
        for side, dd in self.saved["tuned"].items():
            for direction, t in dd.items():
                self.assertEqual(t["n_val"], int(n_val[side]))
                self.assertTrue(((t["rho"] >= 0) & (t["rho"] <= 1)).all())
                v = self.p[(self.p.cf_split == "pair_val") & (self.p.side == side)].reset_index(drop=True)
                d = cf3.predict(self.saved["model"], v, self.meas)[direction]
                np.testing.assert_allclose(cf3.fit_rho(d), t["rho"])

    def test_tau_sign_agrees_with_phase5_att(self):
        att = self.summary["consistency_with_phase5_att"]
        agree = [np.sign(v["track_a_mean_tau_sd"]) == np.sign(v["phase5_att_sd"]) for v in att.values()]
        self.assertEqual(sum(agree), 16)

    def test_no_identifiers_in_committed_outputs(self):
        from local_mcp import egress
        from roadmap.config import SEED
        for name in ("cf3_01_track_a.json", "CF3_PAIR_VALIDATION.csv", "CF3_SUBGROUPS.csv"):
            text = (DOC / name).read_text().replace(str(SEED), "<seed>")
            self.assertEqual(egress.scan_text(text), [], name)


if __name__ == "__main__":
    unittest.main()
