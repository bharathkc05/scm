"""Patient explorer tests. Assertions never print patient values (outputs of these tests may be shared).

    python -X utf8 -m unittest roadmap.phase10_counterfactual_ecg.test_patient_explorer -v
"""
import subprocess
import unittest

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, REPO
from roadmap.phase10_counterfactual_ecg import cf3_01_track_a as A
from roadmap.phase10_counterfactual_ecg import cf4_model as B
from roadmap.phase10_counterfactual_ecg import patient_explorer as X

OUT = DATA_OUT / "phase10"


class Explorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = X.Context()
        cls.picks = {k: [X.choose(cls.ctx, k, s, False) for s in range(5)] for k in ("pair", "mi", "control")}

    def test_cf4_training_rows_unchanged_by_refactor(self):
        saved = pd.read_parquet(OUT / "cf4_rows.parquet")
        now = B.build_rows()
        self.assertTrue(np.array_equal(saved.record_id.to_numpy(), now.record_id.to_numpy()))
        self.assertTrue(np.allclose(saved[B.COVS].to_numpy(float), now[B.COVS].to_numpy(float)))

    def test_random_selection_is_held_out_and_kind_matches(self):
        for kind, sids in self.picks.items():
            for sid in sids:
                u = X.training_use(self.ctx, sid)
                self.assertFalse(u["track_a_trained_on_patient"] or u["track_b_trained_on_patient"], kind)
                self.assertFalse(u["track_a_rho_and_intervals_tuned_on_patient"], kind)
                self.assertEqual(X.kind_of(self.ctx, sid), kind)

    def test_pair_view_reproduces_cf3_validation(self):
        pred = pd.read_parquet(OUT / "cf3_predictions.parquet")
        pred = pred[pred.side == "before"]
        p = A.pair_table(self.ctx.meas, self.ctx.ep, self.ctx.pairs, self.ctx.split)
        prim = p[p.primary & (p.cf_split == "pair_test")].sample(5, random_state=1)
        for _, r in prim.iterrows():
            prow = prim[(prim.hadm_id == r.hadm_id) & (prim.side == r.side)]
            for direction in ("add", "remove"):
                ta = X.track_a_pair(self.ctx, prow, direction)
                ref = pred[(pred.hadm_id == r.hadm_id) & (pred.direction == direction)]
                self.assertEqual(len(ref), 1)
                cols = [f"A2_shrunk__{m}" for m in A.ROBUST]
                self.assertTrue(np.allclose(ta.counterfactual.to_numpy(), ref[cols].to_numpy()[0], equal_nan=True))

    def test_track_b_same_state_returns_observed(self):
        sid = self.picks["mi"][0]
        row = self.ctx.rows[self.ctx.rows.subject_id == sid].iloc[0]
        tb = X.track_b(self.ctx, row, row.mi_state)
        self.assertTrue(np.allclose(tb["counterfactual"], tb["observed"], atol=1e-5))
        self.assertEqual(tb["counterfactual"].shape, (12, 400))

    def test_track_a_single_directions(self):
        rows = self.ctx.rows
        mi_row = rows[(rows.subject_id == self.picks["mi"][0]) & (rows.mi_state != "none")].iloc[0]
        ctrl_row = rows[(rows.subject_id == self.picks["control"][0]) & (rows.role == "control")].iloc[0]
        rem = X.track_a_single(self.ctx, mi_row, "remove", "none")
        add = X.track_a_single(self.ctx, ctrl_row, "add", "stemi_anterior")
        for t in (rem, add):
            self.assertEqual(list(t.measure), A.ROBUST)
            ok = t.counterfactual.notna()
            self.assertTrue((t.low[ok] < t.counterfactual[ok]).all() and (t.counterfactual[ok] < t.high[ok]).all())

    def test_clinical_risks_and_pn(self):
        for kind in ("pair", "control"):
            sid = self.picks[kind][0]
            rows = self.ctx.rows[self.ctx.rows.subject_id == sid]
            hadm = self.ctx.ep[self.ctx.ep.subject_id == sid].hadm_id.iat[0] if kind == "pair" else None
            cl = X.clinical(self.ctx, X.profile_for(self.ctx, sid, kind, hadm, rows.iloc[0]), kind == "pair")
            self.assertEqual(len(cl), 3)
            self.assertTrue(((cl.risk_do_no > 0) & (cl.risk_do_no < 1) & (cl.risk_do_yes > 0)
                             & (cl.risk_do_yes < 1)).all())
            pn = cl.pn.dropna()
            self.assertTrue(((pn >= 0) & (pn <= 1)).all())
            if kind == "control":
                self.assertTrue(cl.pn.isna().all(), "PN only for MI patients")

    def test_reports_render_for_each_kind(self):
        heads = {"pair": "checked against the patient's real ECGs", "mi": "Without MI", "control": "With MI"}
        for kind, sids in self.picks.items():
            name, doc = X.report(self.ctx, sids[0])
            self.assertIn(heads[kind], doc)
            self.assertIn("Clinical interventions", doc)
            self.assertIn("data:image/png;base64", doc)
            self.assertNotIn(str(sids[0]), name, "file name must not contain the subject id")

    def test_report_folder_is_git_ignored(self):
        r = subprocess.run(["git", "check-ignore", "-q", str(X.REPORTS / "report_x.html")], cwd=REPO)
        self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
