/* ECG explainer core (Phase 9). Runs in the browser and in Node (tests).
 * Compares one ECG with what the fitted Layer 2 model expects for the same patient WITHOUT MI, and shows the
 * difference beside the estimated average MI effect. It never outputs a probability, a diagnosis or advice.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.ECGExplainer = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const CAVEAT = "Research prototype: not a diagnosis, not medical advice. The expected ECG is a model-implied " +
    "comparison for a similar patient without MI, not this patient's true counterfactual. Effects hold only " +
    "under the study's stated assumptions (hospitalized adults, ECG recorded after MI onset).";

  // ---- B-spline basis (Cox-de Boor), constant extrapolation = clamp to the boundary knots ----
  function bsplineAll(x, t, k) {
    const n = t.length - k - 1;
    const lo = t[k], hi = t[n];
    let xc = Math.min(Math.max(x, lo), hi);
    if (xc >= hi) xc = hi - 1e-12 * Math.max(1, Math.abs(hi));      // right end: use the left limit
    let B = new Array(t.length - 1).fill(0);
    for (let i = 0; i < t.length - 1; i++) if (t[i] <= xc && xc < t[i + 1]) B[i] = 1;
    for (let d = 1; d <= k; d++) {
      const next = new Array(t.length - 1 - d).fill(0);
      for (let i = 0; i < next.length; i++) {
        const a = t[i + d] - t[i], b = t[i + d + 1] - t[i + 1];
        next[i] = (a > 0 ? (xc - t[i]) / a * B[i] : 0) + (b > 0 ? (t[i + d + 1] - xc) / b * B[i + 1] : 0);
      }
      B = next;
    }
    return B.slice(0, n);
  }

  // ---- design row (must match roadmap/estimators.py Design, spec age_df5_int_agexall) ----
  function designRow(patient, S) {
    const full = bsplineAll(patient.age, S.age_spline.knots, S.age_spline.degree);
    const age = S.age_spline.kept_columns.map((j) => full[j]);
    const b = S.covariates.map((c) => Number(patient[c]));
    const row = [...age, ...b];
    for (let i = 0; i < b.length; i++) for (const a of age) row.push(a * b[i]);
    for (let i = 0; i < b.length; i++) for (let j = i + 1; j < b.length; j++) row.push(b[i] * b[j]);
    if (row.length !== S.scaler_mean.length) throw new Error("design length mismatch");
    return row.map((v, i) => (v - S.scaler_mean[i]) / S.scaler_scale[i]);
  }

  function expectedWithoutMI(patient, P) {
    const S = P.structural, x = designRow(patient, S), out = {};
    S.measures.forEach((m, j) => {
      let v = S.intercept[j];
      for (let i = 0; i < x.length; i++) v += S.coef[j][i] * x[i];
      out[m] = v;
    });
    return out;
  }

  // ---- input handling ----
  function swapLeadLabels(features) {           // raw ECGdeli files label aVL/aVF swapped
    const out = {};
    for (const [k, v] of Object.entries(features)) {
      const nk = k.includes("_aVL") ? k.replace("_aVL", "_aVF") : k.includes("_aVF") ? k.replace("_aVF", "_aVL") : k;
      out[nk] = v;
    }
    return out;
  }

  function winsorize(v, bounds) { return Math.min(Math.max(v, bounds[0]), bounds[1]); }

  function validate(patient, features, P) {
    const errors = [], outside = [], pre = P.preprocessing;
    const [amin, amax] = pre.age_limits;
    if (!Number.isFinite(Number(patient.age)) || patient.age === "" || patient.age === null) errors.push("Age is missing or not a number.");
    else if (patient.age < amin || patient.age > amax) errors.push(`Age ${patient.age} is outside ${amin}-${amax}.`);
    for (const c of P.structural.covariates) {
      const v = Number(patient[c]);
      if (!(v === 0 || v === 1) || patient[c] === "" || patient[c] === null || patient[c] === undefined) errors.push(`${c} must be 0 or 1.`);
    }
    for (const f of [...pre.point_inputs, pre.count_input]) {
      const raw = features[f];
      if (raw === undefined || raw === null || raw === "") { errors.push(`Missing ECG field: ${f}.`); continue; }
      const v = Number(raw);
      if (!Number.isFinite(v)) { errors.push(`ECG field ${f} is not a number (${raw}).`); continue; }
      const r = pre.training_range_p001_p999[f];
      if (r) {
        const w = r[1] - r[0];
        if (v < r[0] - w || v > r[1] + w) errors.push(`ECG field ${f} = ${v} is far outside the training range (${r[0].toFixed(3)} to ${r[1].toFixed(3)}).`);
        else if (v < r[0] || v > r[1]) outside.push(f);
      }
    }
    if (Number(features.RR_Mean_Global) <= 0) errors.push("RR_Mean_Global must be positive.");
    if (Number(features[pre.count_input]) <= 0) errors.push(`${pre.count_input} must be positive.`);
    return { errors, outside };
  }

  function explain(patient, featuresIn, P, opts = {}) {
    const features = opts.rawEcgdeli ? swapLeadLabels(featuresIn) : { ...featuresIn };
    const { errors, outside } = validate(patient, features, P);
    const result = { caveat: CAVEAT, flags: [], rejected: errors.length > 0, errors, rows: [], groups: [] };
    if (opts.timing !== "after_onset") result.flags.push("Unknown ECG timing: effects apply to ECGs recorded after MI onset.");
    if (result.rejected) return result;

    const pre = P.preprocessing, f = {};
    for (const k of pre.point_inputs) f[k] = Number(features[k]);
    const count = Number(features[pre.count_input]);
    const ratio = count / (10000 / f.RR_Mean_Global);
    if (count < pre.beat_count_min || ratio < pre.beat_ratio_limits[0] || ratio > pre.beat_ratio_limits[1])
      result.flags.push(`Low or inconsistent beat count (${count} beats detected; ${(ratio * 100).toFixed(0)}% of expected).`);
    if (outside.length) result.flags.push(`Inputs outside the range seen in training: ${outside.join(", ")}.`);

    const comp = pre.composites;
    const obsRaw = { ...f,
      ST_lateral_mean: comp.ST_lateral_mean.mean_of.reduce((s, k) => s + f[k], 0) / comp.ST_lateral_mean.mean_of.length,
      ST_max_depression: Math.min(...comp.ST_max_depression.min_of.map((k) => f[k])) };
    const expected = expectedWithoutMI(patient, P);
    for (const m of P.structural.measures) {
      const e = P.effects[m];
      const observed = winsorize(obsRaw[m], pre.winsor_bounds[m]);
      const diff = observed - expected[m];
      result.rows.push({ measure: m, group: e.group, observed, expected_without_mi: expected[m], difference: diff,
        difference_sd: diff / e.control_sd, mi_effect: e.att, mi_effect_ci: e.att_ci, mi_effect_sd: e.att_sd,
        in_mi_effect_direction: Math.sign(diff) === Math.sign(e.att) });
    }
    for (const g of ["ST", "T", "Q"]) {
      const rs = result.rows.filter((r) => r.group === g);
      result.groups.push({ group: g, measures: rs.length, in_mi_direction: rs.filter((r) => r.in_mi_effect_direction).length,
        mean_aligned_difference_sd: rs.reduce((s, r) => s + Math.sign(r.mi_effect) * r.difference_sd, 0) / rs.length });
    }
    return result;
  }

  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter((l) => l.trim() !== "");
    if (lines.length < 2) throw new Error("The file needs a header row and at least one data row.");
    const header = lines[0].split(",").map((h) => h.trim());
    return lines.slice(1).map((l) => {
      const v = l.split(","), row = {};
      header.forEach((h, i) => { row[h] = v[i] === undefined ? "" : v[i].trim(); });
      return row;
    });
  }

  return { CAVEAT, bsplineAll, designRow, expectedWithoutMI, swapLeadLabels, winsorize, validate, explain, parseCSV };
});
