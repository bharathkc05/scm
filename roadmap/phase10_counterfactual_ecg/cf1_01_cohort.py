"""Phase 10, CF1 — MI episodes, ECG index, within-patient pairs and the Phase 10 split.

Inputs (read-only): Phase 2 admissions (flags, covariates, ED arrival), Phase 3 patient splits, and the MIMIC
DuckDB copy (diagnoses_icd, procedures_icd, labevents, record_list, patients). All joins run in DuckDB.

Outputs (git-ignored, patient-level), data/roadmap/phase10/:
  episodes.parquet   one row per acute-MI admission: kind, territory, timing context, troponin anchor, revascularisation
  ecg_index.parquet  every ECG of patients with an MI episode (role: during / baseline_before / baseline_after / other)
                     plus one ECG per never-MI patient (role: control), with age and covariates at the ECG
  pairs.parquet      one row per episode with a during-MI ECG and at least one baseline ECG
  cf_splits.parquet  one row per patient: Phase 10 split
Summary (aggregate, committed): roadmap/phase10_counterfactual_ecg/cf1_01_cohort.json
"""
from __future__ import annotations

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS, SEED
from roadmap.definitions import (CF_PAIR_SHARES, MI_KIND_PRIORITY, PAIR_AFTER_GAP_DAYS, PAIR_BEFORE_GAP_HOURS,
                                 REVASC_CODES, TERRITORY_PRIORITY, TROPONIN_ITEMIDS, TROPONIN_LOOKBACK_HOURS,
                                 mi_territory)

OUT = DATA_OUT / "phase10"
SUMMARY = PHASE_DIRS[10] / "cf1_01_cohort.json"
MIN_CELL = 11


def small(n: int) -> int | str:
    """Aggregate counts below 11 are not published (same rule as the MCP server)."""
    return int(n) if n >= MIN_CELL else f"<{MIN_CELL}"


def register(con, name: str, df: pd.DataFrame) -> None:
    """DuckDB cannot read pandas' string dtype; hand it plain object columns."""
    con.register(name, df.astype({c: object for c in df.columns if pd.api.types.is_string_dtype(df[c])}))


def territories(con) -> pd.DataFrame:
    """Kind and territory per acute-MI admission, from every MI code on the admission."""
    dx = con.sql("""select d.hadm_id, d.icd_code, d.icd_version from diagnoses_icd d
                    join adm a using (hadm_id) where a.acute_mi = 1""").df()
    rows = []
    for hadm, g in dx.groupby("hadm_id"):
        terr = [t for t in (mi_territory(c, v) for c, v in zip(g.icd_code, g.icd_version)) if t]
        if not terr:
            continue
        coarse = {c for _, c in terr}
        stemi = [c for c in coarse if c in ("anterior", "inferior", "lateral", "posterior", "other_site")]
        best = min(coarse, key=TERRITORY_PRIORITY.index)
        rows.append({"hadm_id": hadm, "territory": best,
                     "territory_detail": "+".join(sorted({d for d, c in terr if c == best})),
                     "multi_territory": len(set(stemi)) > 1})
    return pd.DataFrame(rows)


def main() -> dict:
    con = duckdb.connect(str(PATHS["mimic_duckdb"]), read_only=True)
    con.execute("SET threads = 6")
    adm = pd.read_parquet(DATA_OUT / "phase2/admissions.parquet", columns=[
        "subject_id", "hadm_id", "admittime", "dischtime", "arrival_time", "planned_admission", "acute_mi",
        "acute_mi_principal", "mi_stemi", "mi_nstemi", "mi_type2", "mi_other_unspecified", "age", "sex_male",
        "diabetes_primary", "hypertension_primary", "ckd_primary", "prior_mi_primary"])
    register(con, "adm", adm)
    register(con, "terr", territories(con))

    # ---------------------------------------------------------------- episodes
    revasc = " or ".join([f"(icd_version = 9 and icd_code like '{c}%')" for c in REVASC_CODES[9]]
                         + [f"(icd_version = 10 and icd_code like '{c}%')" for c in REVASC_CODES[10]])
    trop = ",".join(map(str, TROPONIN_ITEMIDS))
    con.execute(f"""
        create temp table ep as
        with base as (
          select a.*, t.territory, t.territory_detail, coalesce(t.multi_territory, false) multi_territory,
                 case when a.mi_stemi = 1 then 'stemi' when a.mi_nstemi = 1 then 'nstemi'
                      when a.mi_type2 = 1 then 'type2' else 'other_unspecified' end as mi_kind,
                 (a.acute_mi_principal = 1 and a.planned_admission = 0) as after_onset_context,
                 row_number() over (partition by a.subject_id order by a.arrival_time) as episode_no
          from adm a left join terr t using (hadm_id) where a.acute_mi = 1),
        rv as (
          select hadm_id, min(chartdate) first_revasc_date from procedures_icd where {revasc} group by 1),
        tr as (
          select b.hadm_id,
                 min(l.charttime) as first_troponin_time,
                 min(l.charttime) filter (where (l.valuenum > l.ref_range_upper and l.ref_range_upper is not null)
                                              or (l.ref_range_upper is null and l.flag = 'abnormal')) as troponin_rise_time
          from base b join labevents l
            on l.subject_id = b.subject_id
           and l.charttime between b.arrival_time - interval {TROPONIN_LOOKBACK_HOURS} hour and b.dischtime
          where l.itemid in ({trop}) group by 1)
        select b.*, rv.first_revasc_date is not null as revascularised,
               date_diff('day', cast(b.arrival_time as date), rv.first_revasc_date) as revasc_day,
               date_diff('minute', b.arrival_time, tr.troponin_rise_time) / 60.0 as troponin_rise_h_from_arrival,
               tr.first_troponin_time is not null as troponin_tested
        from base b left join rv using (hadm_id) left join tr using (hadm_id)""")

    # ------------------------------------------------------------- ECG index
    con.execute("""
        create temp table mi_pat as select distinct subject_id from ep;
        create temp table ecg as
        select r.subject_id, r.study_id, r.ecg_time, r.path,
               p.anchor_age + (year(r.ecg_time) - p.anchor_year) as age_at_ecg, (p.gender = 'M') as sex_male
        from record_list r join patients p using (subject_id);""")
    # Which MI stay (if any) an ECG falls in, and the patient's MI windows for baseline rules.
    con.execute(f"""
        create temp table mi_ecg as
        select e.*, ep.hadm_id as during_hadm_id,
               date_diff('minute', ep.arrival_time, e.ecg_time) / 60.0 as hours_from_arrival,
               case when ep.troponin_rise_h_from_arrival is null then null
                    else date_diff('minute', ep.arrival_time, e.ecg_time) / 60.0 - ep.troponin_rise_h_from_arrival
               end as hours_from_troponin_rise
        from ecg e join mi_pat using (subject_id)
        left join ep on ep.subject_id = e.subject_id and e.ecg_time between ep.arrival_time and ep.dischtime
        -- two MI admissions can share one ED arrival (discharge and readmission in the same visit):
        -- an ECG belongs to the earliest of them
        qualify row_number() over (partition by e.study_id order by ep.admittime nulls last) = 1""")
    # Covariates at the ECG: the most recent admission that started at or before it (as-of join).
    con.execute("""
        create temp table mi_ecg_cov as
        select m.*, a.hadm_id as asof_hadm_id, a.diabetes_primary as diabetes, a.hypertension_primary as hypertension,
               a.ckd_primary as ckd, a.prior_mi_primary as prior_mi
        from mi_ecg m asof left join adm a on m.subject_id = a.subject_id and a.admittime <= m.ecg_time""")

    # ---------------------------------------------------------------- pairs
    con.execute(f"""
        create temp table pairs as
        with during as (
          select during_hadm_id hadm_id, count(*) n_during_ecgs,
                 arg_min(study_id, ecg_time) during_study_id, min(ecg_time) during_ecg_time,
                 min(hours_from_arrival) during_hours_from_arrival
          from mi_ecg where during_hadm_id is not null group by 1),
        outside as (select * from mi_ecg where during_hadm_id is null),
        before as (
          select ep.hadm_id, arg_max(o.study_id, o.ecg_time) before_study_id, max(o.ecg_time) before_ecg_time
          from ep join outside o on o.subject_id = ep.subject_id
           and o.ecg_time < ep.arrival_time - interval {PAIR_BEFORE_GAP_HOURS} hour
          group by 1),
        nxt as (
          select hadm_id, lead(arrival_time) over (partition by subject_id order by arrival_time) next_mi_arrival from ep),
        after as (
          select ep.hadm_id, arg_min(o.study_id, o.ecg_time) after_study_id, min(o.ecg_time) after_ecg_time
          from ep join nxt using (hadm_id) join outside o on o.subject_id = ep.subject_id
           and o.ecg_time > ep.dischtime + interval {PAIR_AFTER_GAP_DAYS} day
           and (nxt.next_mi_arrival is null or o.ecg_time < nxt.next_mi_arrival - interval 1 day)
          group by 1)
        select ep.subject_id, ep.hadm_id, ep.episode_no, ep.mi_kind, ep.territory, ep.after_onset_context,
               d.*, b.before_study_id, b.before_ecg_time, a.after_study_id, a.after_ecg_time,
               date_diff('hour', b.before_ecg_time, d.during_ecg_time) / 24.0 as before_gap_days,
               date_diff('hour', d.during_ecg_time, a.after_ecg_time) / 24.0 as after_gap_days,
               exists (select 1 from ep e2 where e2.subject_id = ep.subject_id and e2.dischtime < b.before_ecg_time)
                 as before_follows_earlier_mi
        from ep join during d using (hadm_id) left join before b using (hadm_id) left join after a using (hadm_id)
        where b.before_study_id is not null or a.after_study_id is not null""")

    # ---------------------------------------------------------------- roles
    con.execute("""
        create temp table roles as
        select study_id, 'during' as role from mi_ecg where during_hadm_id is not null
        union all select before_study_id, 'baseline_before' from pairs where before_study_id is not null
        union all select after_study_id, 'baseline_after' from pairs where after_study_id is not null;
        create temp table ecg_index as
        select m.subject_id, m.study_id, m.ecg_time, m.path, m.age_at_ecg, m.sex_male, m.during_hadm_id,
               m.hours_from_arrival, m.hours_from_troponin_rise, m.diabetes, m.hypertension, m.ckd, m.prior_mi,
               coalesce((select string_agg(distinct r.role, '+' order by r.role) from roles r where r.study_id = m.study_id),
                        'other_mi_patient') as role,
               m.study_id in (select during_study_id from pairs) as is_pair_during
        from mi_ecg_cov m
        union all
        select * exclude (rn) from (
          select e.subject_id, e.study_id, e.ecg_time, e.path, e.age_at_ecg, e.sex_male, null, null, null,
                 a.diabetes_primary, a.hypertension_primary, a.ckd_primary, a.prior_mi_primary, 'control', false,
                 row_number() over (partition by e.subject_id order by e.ecg_time) rn
          from ecg e asof left join adm a on e.subject_id = a.subject_id and a.admittime <= e.ecg_time
          where e.subject_id not in (select subject_id from mi_pat) and e.age_at_ecg >= 18) where rn = 1""")

    episodes, ecg_index, pairs = (con.sql(f"select * from {t}").df() for t in ("ep", "ecg_index", "pairs"))

    # ---------------------------------------------------------------- splits
    sp = pd.read_parquet(DATA_OUT / "phase3/patient_splits.parquet")[["subject_id", "split"]]
    first_pair = pairs.sort_values(["subject_id", "episode_no"]).drop_duplicates("subject_id")[["subject_id", "mi_kind"]]
    rng = np.random.default_rng(SEED)
    assign = {}
    for kind, g in first_pair.groupby("mi_kind"):
        ids = np.sort(g.subject_id.to_numpy())
        rng.shuffle(ids)
        n_test = int(round(len(ids) * CF_PAIR_SHARES["pair_test"]))
        assign.update({int(s): ("pair_test" if i < n_test else "pair_val") for i, s in enumerate(ids)})
    sp["cf_split"] = sp.subject_id.map(assign).fillna(sp.split)
    sp["phase3_split"] = sp.pop("split")
    digest = hashlib.sha256(sp.sort_values("subject_id")[["subject_id", "cf_split"]]
                            .to_csv(index=False).encode()).hexdigest()

    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in (("episodes", episodes), ("ecg_index", ecg_index), ("pairs", pairs), ("cf_splits", sp)):
        df.to_parquet(OUT / f"{name}.parquet", index=False)

    # ---------------------------------------------------------------- summary (aggregates only)
    pairs = pairs.merge(sp[["subject_id", "cf_split"]], on="subject_id")
    q = lambda s: [round(float(x), 1) for x in s.quantile([0.25, 0.5, 0.75])] if len(s) else None
    roles = ecg_index.role.str.split("+").explode().value_counts()
    s = {
        "episodes": {"acute_mi_admissions": len(episodes), "patients": int(episodes.subject_id.nunique()),
                     "by_kind": {k: small(v) for k, v in episodes.mi_kind.value_counts().items()},
                     "stemi_by_territory": {k: small(v) for k, v in episodes[episodes.mi_kind == "stemi"]
                                            .territory.value_counts(dropna=False).items()},
                     "multi_territory_stemi": small(int(episodes.multi_territory.sum())),
                     "after_onset_context": small(int(episodes.after_onset_context.sum())),
                     "troponin_tested_share": round(float(episodes.troponin_tested.mean()), 3),
                     "troponin_rise_found_share": round(float(episodes.troponin_rise_h_from_arrival.notna().mean()), 3),
                     "troponin_rise_h_from_arrival_iqr": q(episodes.troponin_rise_h_from_arrival.dropna()),
                     "revascularised_share": round(float(episodes.revascularised.mean()), 3)},
        "ecg_index": {"ecgs": len(ecg_index), "by_role": {k: small(v) for k, v in roles.items()},
                      "mi_patients_with_ecg": int(ecg_index[ecg_index.role != "control"].subject_id.nunique()),
                      "control_patients": int((ecg_index.role == "control").sum()),
                      "during_ecgs_hours_from_arrival_iqr": q(ecg_index.hours_from_arrival.dropna())},
        "pairs": {"episodes": len(pairs), "patients": int(pairs.subject_id.nunique()),
                  "with_before": small(int(pairs.before_study_id.notna().sum())),
                  "with_after": small(int(pairs.after_study_id.notna().sum())),
                  "with_both": small(int((pairs.before_study_id.notna() & pairs.after_study_id.notna()).sum())),
                  "first_episode": small(int((pairs.episode_no == 1).sum())),
                  "after_onset_context": small(int(pairs.after_onset_context.sum())),
                  "before_follows_earlier_mi": small(int(pairs.before_follows_earlier_mi.sum())),
                  "by_kind": {k: small(v) for k, v in pairs.mi_kind.value_counts().items()},
                  "stemi_by_territory": {k: small(v) for k, v in pairs[pairs.mi_kind == "stemi"]
                                         .territory.value_counts(dropna=False).items()},
                  "before_gap_days_iqr": q(pairs.before_gap_days.dropna()),
                  "after_gap_days_iqr": q(pairs.after_gap_days.dropna())},
        "splits": {"rule": "pair patients 80% pair_test / 20% pair_val by MI kind of their first pair episode; "
                           "everyone else keeps the Phase 3 split",
                   "patients_by_cf_split": sp.cf_split.value_counts().to_dict(),
                   "pair_patients_by_cf_split": pairs.drop_duplicates("subject_id").cf_split.value_counts().to_dict(),
                   "pair_patients_moved_out_of_phase3_train": int(sp[sp.cf_split.str.startswith("pair")]
                                                                  .phase3_split.eq("train").sum()),
                   "sha256": digest, "seed": SEED},
        "gate_cf_g0": None,
    }
    n_test = s["splits"]["pair_patients_by_cf_split"].get("pair_test", 0)
    s["gate_cf_g0"] = {"rule": "at least 3,000 pair patients in pair_test", "pair_test_patients": n_test,
                       "pass": n_test >= 3000}
    return s


if __name__ == "__main__":
    s = main()
    SUMMARY.write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))
