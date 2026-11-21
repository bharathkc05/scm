# ECG Explainer — user note

**Research prototype: not a diagnosis, not medical advice.** It does not compute a probability of MI and
gives no treatment or lifestyle advice.

## What it does

You enter six baseline facts (age, sex, diabetes, hypertension, CKD, prior MI) and one ECG's features. The
page shows, for the 16 ECG measures that passed every check in the study (ST level in I, aVL, V4–V6, III,
aVR; T amplitude in V4–V6, aVR, II; Q amplitude in I, aVL; lateral ST mean; maximum ST depression):

- the **observed** value (winsorized to the training range);
- the value the study's model **expects for a similar patient without MI**;
- the **difference**, in control standard deviations;
- the study's **estimated average MI effect** on that measure, with its 95% interval;
- whether the difference points the same way as the MI effect, summarized per group (ST, T, Q).

The expected value is a model-implied comparison, not this patient's true counterfactual. The MI effect is a
population average among MI admissions, not an estimate for this patient. A single ECG differing in the
MI direction does not indicate MI: in the study only 20–25% of MI patients differed by more than 1 SD.

## How to use it

1. Open `ecg_explainer.html` in a browser. Everything runs locally; no data leaves the computer.
2. Enter the patient.
3. Choose the ECG timing: the study's effects apply to ECGs recorded after MI onset.
4. Provide the ECG features in one of two ways:
   - **Load a CSV** in the training format: the 53 ECGdeli point features plus `QRS_Dur_Global_count`;
     other columns are ignored.
   - **Fill with training medians** for an example, then edit the fields.
5. If the file comes straight from the project's MATLAB/ECGdeli extraction, keep **"raw ECGdeli output"**
   ticked. That pipeline labels aVL and aVF the wrong way round, and the page swaps them back.
6. Press **Run comparison**.

## Flags and rejections

- **Rejected, with a message:** missing or non-numeric fields; age outside 18–100; history values other than
  0/1; ECG values far outside the training range.
- **Flagged:**
  - unknown ECG timing;
  - a low or inconsistent beat count;
  - values outside the 0.1–99.9th percentile of training (these are clipped).
- The caveat is shown with every result.

The training-median example is not a real ECG. Maximum ST depression is the minimum across leads, and
the minimum of the median leads is not a typical value, so that row looks unusual in the example.

## Files

| File | Role |
|---|---|
| `ecg_explainer.html` | The page (self-contained: parameters and code inlined) |
| `explainer.js` | Core logic, shared by the page and the tests |
| `demo_params.json` | Exported parameters: no-MI model, training-split preprocessing, effects with intervals |
| `p9_01_export.py`, `p9_02_build_page.py` | Rebuild the parameters and the page |
| `tests/` | Acceptance tests (`python -m unittest roadmap.phase9_demo.tests.test_demo -v`) |
