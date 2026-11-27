"""CF4 acceptance tests: Track B data separation, conditioning, model mechanics and committed outputs.

    python -X utf8 -m unittest roadmap.phase10_counterfactual_ecg.test_cf4 -v
"""
import json
import unittest

import h5py
import numpy as np
import pandas as pd
import torch

from roadmap.config import DATA_OUT, PHASE_DIRS, REPO
from roadmap.phase10_counterfactual_ecg import cf4_model as M

OUT = DATA_OUT / "phase10"
DOC = PHASE_DIRS[10]


class Data(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = M.build_rows()
        cls.pairs = pd.read_parquet(OUT / "pairs.parquet")

    def test_no_pair_patient_and_only_model_splits(self):
        self.assertFalse(set(self.rows.subject_id) & set(self.pairs.subject_id))
        self.assertTrue(self.rows.split.isin(["train", "validation", "test"]).all())
        per_patient = self.rows.groupby("subject_id").split.nunique()
        self.assertTrue((per_patient == 1).all(), "a patient in two splits")

    def test_mi_state_follows_role(self):
        during = self.rows.role.str.contains("during")
        self.assertTrue((self.rows.loc[~during, "mi_state"] == "none").all())
        self.assertTrue(self.rows.loc[during, "mi_state"].isin(M.MI_STATES[1:]).all())

    def test_condition_matrix(self):
        c = M.condition(self.rows.head(1000))
        self.assertEqual(c.shape[1], M.C_DIM)
        np.testing.assert_array_equal(c[:, :len(M.MI_STATES)].sum(1), 1)
        cf = M.condition(self.rows.head(1000), "stemi_anterior")
        self.assertTrue((cf[:, M.MI_STATES.index("stemi_anterior")] == 1).all())
        np.testing.assert_array_equal(cf[:, len(M.MI_STATES):], c[:, len(M.MI_STATES):])   # only the MI state moves

    def test_load_beats_matches_h5(self):
        rows = np.sort(self.rows.row.sample(50, random_state=0).to_numpy())
        x = M.load_beats(rows)
        with h5py.File(OUT / "beats.h5", "r") as h:
            ref = np.stack([h["beats"][r] for r in rows]).astype(np.float32)
        np.testing.assert_array_equal(x, ref)


class Model(unittest.TestCase):
    def test_shapes_and_identity_counterfactual(self):
        torch.manual_seed(0)
        m = M.CondVAE().eval()
        x = torch.randn(4, M.N_LEADS, M.N_SAMPLES)
        c = torch.zeros(4, M.C_DIM)
        c[:, 0] = 1
        xh, mu, lv, logits = m(x, c)
        self.assertEqual(tuple(xh.shape), (4, M.N_LEADS, M.N_SAMPLES))
        self.assertEqual(tuple(logits.shape), (4, len(M.MI_STATES)))
        torch.testing.assert_close(m.counterfactual(x, c, c), xh)   # same condition = reconstruction (eval: z = mu)

    def test_gradient_reversal(self):
        z = torch.ones(3, requires_grad=True)
        M.GradReverse.apply(z, 2.0).sum().backward()
        torch.testing.assert_close(z.grad, torch.full((3,), -2.0))

    def test_fvu(self):
        x = np.random.default_rng(0).normal(size=(5, 12, 400))
        self.assertAlmostEqual(M.fvu_per_lead(x, x), 0.0)
        self.assertAlmostEqual(M.fvu_per_lead(x, x.mean(-1, keepdims=True) + 0 * x), 1.0)


class Outputs(unittest.TestCase):
    def test_selected_checkpoint_loads(self):
        ck = torch.load(REPO / "models/cf_vae/cvae_selected.pt", map_location="cpu")
        m = M.CondVAE(ck["z_dim"])
        m.load_state_dict(ck["state_dict"])
        self.assertEqual(ck["mi_states"], M.MI_STATES)

    def test_summary_consistent_with_sweep(self):
        s = json.loads((DOC / "cf4_01_train.json").read_text())
        sweep = pd.read_csv(DOC / "CF4_SWEEP.csv")
        self.assertIn(s["chosen"], set(sweep.run))
        ok = sweep[sweep.auroc_max <= 0.55]
        expect = (ok.sort_values("val_fvu_per_lead") if len(ok) else sweep.sort_values("auroc_max")).iloc[0].run
        self.assertEqual(s["chosen"], expect, "selection rule")

    def test_no_identifiers_in_committed_outputs(self):
        from local_mcp import egress
        from roadmap.config import SEED
        for name in ("cf4_01_train.json", "CF4_SWEEP.csv"):
            text = (DOC / name).read_text().replace(str(SEED), "<seed>")
            self.assertEqual(egress.scan_text(text), [], name)


if __name__ == "__main__":
    unittest.main()
