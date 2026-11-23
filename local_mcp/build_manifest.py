"""Build local_mcp/manifest.json: every file the server may read, with tier, hash, size and DuckDB view.

Run from the repo root after re-running any roadmap phase:
    <project python> -X utf8 local_mcp/build_manifest.py

Text files (csv/json/sql) are hashed with CRLF normalised to LF, because git on Windows may check them out
with different line endings than they were written with. Raw MIMIC files (tier T0) can never be registered.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_mcp.paths import MANIFEST, ROOT, SQL_DIR  # noqa: E402

TEXT_KINDS = {"csv", "json", "sql"}

# (repo-relative path, tier, kind, DuckDB view name or None)
ARTIFACTS = [
    # T1 patient-level analysis data (git-ignored; possession-gated)
    ("data/roadmap/phase3/layer1.parquet", "T1", "parquet", "layer1"),
    ("data/roadmap/phase3/layer2.parquet", "T1", "parquet", "layer2"),
    # T2 aggregate results
    ("roadmap/phase3_dataset/cohort_flow.csv", "T2", "csv", "cohort_flow"),
    ("roadmap/phase4_graph/ADJUSTMENT_TABLE.csv", "T2", "csv", "adjustment_table"),
    ("roadmap/phase4_graph/INDEPENDENCE_TESTS.csv", "T2", "csv", "independence_tests"),
    ("roadmap/phase5_estimation/LAYER1_ESTIMATES.csv", "T2", "csv", "layer1_estimates"),
    ("roadmap/phase5_estimation/LAYER1_DIAGNOSTICS.csv", "T2", "csv", "layer1_diagnostics"),
    ("roadmap/phase5_estimation/LAYER2_ESTIMATES.csv", "T2", "csv", "layer2_estimates"),
    ("roadmap/phase5_estimation/LAYER2_TEST_STABILITY.csv", "T2", "csv", "layer2_test_stability"),
    ("roadmap/phase6_novelty/SENSITIVITY_ESTIMATES.csv", "T2", "csv", "sensitivity_estimates"),
    ("roadmap/phase6_novelty/DECONFOUNDED_ECG_EFFECTS.csv", "T2", "csv", "deconfounded_ecg_effects"),
    ("roadmap/phase6_novelty/FEATURE_GROUP_SUMMARY.csv", "T2", "csv", "feature_group_summary"),
    ("roadmap/phase6_novelty/LATENT_AUDIT.csv", "T2", "csv", "latent_audit"),
    ("roadmap/phase6_novelty/COUNTERFACTUAL_PROFILES.csv", "T2", "csv", "counterfactual_profiles"),
    ("roadmap/phase7_stress_test/EVALUES.csv", "T2", "csv", "evalues"),
    ("roadmap/phase7_stress_test/SELECTION_WEIGHTED_LAYER2.csv", "T2", "csv", "selection_weighted_layer2"),
    ("roadmap/phase7_stress_test/BMI_LAYER2_WITH.csv", "T2", "csv", "bmi_layer2_with"),
    ("roadmap/phase7_stress_test/BMI_LAYER2_WITHOUT.csv", "T2", "csv", "bmi_layer2_without"),
    ("roadmap/phase7_stress_test/OUTCOME_CODES_LAYER1.csv", "T2", "csv", "outcome_codes_layer1"),
    ("roadmap/phase7_stress_test/OUTCOME_CODES_LAYER2_BROAD.csv", "T2", "csv", "outcome_codes_layer2_broad"),
    ("roadmap/phase7_stress_test/SUBGROUPS_LAYER1.csv", "T2", "csv", "subgroups_layer1"),
    ("roadmap/phase7_stress_test/SUBGROUPS_LAYER2.csv", "T2", "csv", "subgroups_layer2"),
    ("roadmap/phase7_stress_test/SUBGROUP_INTERACTIONS_LAYER2.csv", "T2", "csv", "subgroup_interactions_layer2"),
    ("roadmap/phase7_stress_test/LAYER1_HISTORY_VARIANTS.csv", "T2", "csv", "layer1_history_variants"),
    ("roadmap/phase8_paper/CLAIM_EVIDENCE_MAP.csv", "T2", "csv", "claim_evidence_map"),
    ("roadmap/phase8_paper/TABLE1_COHORT.csv", "T2", "csv", "table1_cohort"),
    # T3 model parameters and assets
    ("roadmap/phase9_demo/demo_params.json", "T3", "json", None),
]


def content_hash(path: Path, kind: str) -> str:
    h = hashlib.sha256()
    if kind in TEXT_KINDS:
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    else:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def check_allowed(rel: str, tier: str) -> None:
    """Refuse anything that is not a repo-relative, non-raw file."""
    low = rel.replace("\\", "/").lower()
    if Path(rel).is_absolute() or ":" in rel or ".." in Path(rel).parts:
        raise ValueError(f"{rel}: manifest paths must be relative to the repository")
    if "data/raw" in low or low.endswith(".duckdb"):
        raise ValueError(f"{rel}: raw MIMIC files (tier T0) can never be registered")
    if tier == "T1" and not low.startswith("data/roadmap/"):
        raise ValueError(f"{rel}: T1 files must come from data/roadmap/")


def build(root: Path = ROOT, artifacts=ARTIFACTS, sql_dir: Path = SQL_DIR) -> dict:
    root = Path(root).resolve()
    specs = list(artifacts) + [(p.resolve().relative_to(root).as_posix(), "code", "sql", None)
                               for p in sorted(Path(sql_dir).glob("*.sql"))]
    entries = []
    for rel, tier, kind, view in specs:
        check_allowed(rel, tier)
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"{rel} is missing; run the phase that produces it first")
        st = path.stat()
        entries.append({"path": rel, "tier": tier, "kind": kind, "view": view,
                        "sha256": content_hash(path, kind), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    version = hashlib.sha256(json.dumps([(e["path"], e["sha256"]) for e in entries]).encode()).hexdigest()[:12]
    return {"schema": 1, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": version, "artifacts": entries}


def write(manifest: dict, path: Path = MANIFEST) -> None:
    Path(path).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    m = build()
    write(m)
    by_tier = {}
    for e in m["artifacts"]:
        by_tier[e["tier"]] = by_tier.get(e["tier"], 0) + 1
    print(f"manifest {m['version']}: {len(m['artifacts'])} artifacts {by_tier} -> {MANIFEST}")
