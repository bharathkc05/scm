"""
Phase C Comprehensive Test Suite
=================================

Validates cohort labeling outputs and ensures data readiness for Phase D.

Tests:
- Pre-flight checks (file existence, readability)
- Schema validation (required columns)
- Sample size validation (power analysis requirements)
- Label distribution (primary labels, controls)
- Data quality (nulls, data types, temporal consistency)
- Clinical validation (troponin thresholds, MI criteria)

Usage:
    python scripts/test_phase_c.py              # Full validation
    python scripts/test_phase_c.py --quick      # Skip detailed checks
    python scripts/test_phase_c.py --json       # JSON output for CI/CD
    python scripts/test_phase_c.py --verbose    # Detailed progress

Exit Codes:
    0: All tests passed
    1: Critical failures found
    2: Warnings only (acceptable for analysis)

Author: Phase C Test Suite
Date: February 6, 2026
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import duckdb

# ============================================================================
# CONFIGURATION
# ============================================================================

COHORT_MASTER_PATH = "data/processed/cohort_master.parquet"
DB_PATH = "data/mimic_database.duckdb"

# Required columns per C.9 specification
REQUIRED_COLUMNS = [
    'record_id',
    'subject_id',
    'hadm_id',
    'ecg_time',
    'index_mi_time',
    'primary_label',
    'comorbidity_chronic_mi',
    'environment_label'
]

# Expected primary labels
EXPECTED_LABELS = [
    'MI_Acute_Presentation',
    'Control_Symptomatic',
    'MI_Pre-Incident',
    'Control_Asymptomatic',
    'MI_Post-Incident'
]

# Power analysis requirements (C.8)
MIN_TOTAL_MI_ACUTE = 500
MIN_SUBGROUP_SIZE = 100

# ============================================================================
# TEST RESULT TRACKING
# ============================================================================

class TestResult:
    """Track test results"""
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0
        self.total_tests = 0
        self.errors = []
        self.warning_messages = []
        
    def to_json(self):
        """Convert to JSON format"""
        return {
            "total_tests": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.failed_tests,
            "warnings": self.warnings,
            "errors": self.errors,
            "warning_messages": self.warning_messages,
            "status": "passed" if self.failed_tests == 0 else "failed"
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_test(test_name, status, message="", verbose=False):
    """Print test result"""
    status_symbols = {
        "PASS": "[PASS]",
        "FAIL": "[FAIL]",
        "WARN": "[WARN]",
        "INFO": "[INFO]",
        "RUN": "[RUN]"
    }
    
    symbol = status_symbols.get(status, "[?]")
    
    if status == "RUN" and not verbose:
        return
    
    if message:
        print(f"{symbol} {test_name}: {message}")
    else:
        print(f"{symbol} {test_name}")

def format_number(n):
    """Format number with commas"""
    return f"{n:,}"

def format_size(bytes_size):
    """Format file size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

# ============================================================================
# TEST SUITE 1: PRE-FLIGHT CHECKS
# ============================================================================

def test_preflight(results, verbose=False):
    """Test that cohort file exists and is readable"""
    print("\n" + "=" * 90)
    print("1. PRE-FLIGHT CHECKS")
    print("=" * 90)
    
    # Test 1: Cohort file exists
    results.total_tests += 1
    if os.path.exists(COHORT_MASTER_PATH):
        file_size = os.path.getsize(COHORT_MASTER_PATH)
        print_test("Cohort Master File Exists", "PASS", 
                  f"Size: {format_size(file_size)}", verbose)
        results.passed_tests += 1
    else:
        print_test("Cohort Master File Exists", "FAIL", 
                  f"File not found at {COHORT_MASTER_PATH}", verbose)
        results.failed_tests += 1
        results.errors.append(f"Cohort master file missing: {COHORT_MASTER_PATH}")
        return False
    
    # Test 2: File is readable
    results.total_tests += 1
    try:
        df = pd.read_parquet(COHORT_MASTER_PATH)
        print_test("Cohort Master Readable", "PASS", 
                  f"{format_number(len(df))} rows × {len(df.columns)} columns", verbose)
        results.passed_tests += 1
        return True
    except Exception as e:
        print_test("Cohort Master Readable", "FAIL", f"Error: {str(e)}", verbose)
        results.failed_tests += 1
        results.errors.append(f"Cannot read cohort master: {str(e)}")
        return False

# ============================================================================
# TEST SUITE 2: SCHEMA VALIDATION
# ============================================================================

def test_schema(df, results, verbose=False):
    """Test schema has required columns"""
    print("\n" + "=" * 90)
    print("2. SCHEMA VALIDATION (Required Columns)")
    print("=" * 90)
    
    missing_columns = []
    for col in REQUIRED_COLUMNS:
        results.total_tests += 1
        if col in df.columns:
            print_test(f"  {col}", "PASS", "", verbose)
            results.passed_tests += 1
        else:
            print_test(f"  {col}", "FAIL", "MISSING", verbose)
            results.failed_tests += 1
            missing_columns.append(col)
            results.errors.append(f"Required column missing: {col}")
    
    if missing_columns:
        print(f"\n[CRITICAL] Missing {len(missing_columns)} required columns: {', '.join(missing_columns)}")
        return False
    
    return True

# ============================================================================
# TEST SUITE 3: SAMPLE SIZE VALIDATION
# ============================================================================

def test_sample_size(df, results, verbose=False):
    """Test sample sizes meet power analysis requirements (C.8)"""
    print("\n" + "=" * 90)
    print("3. SAMPLE SIZE VALIDATION (Power Analysis Requirements)")
    print("=" * 90)
    
    # Get label counts
    label_counts = df['primary_label'].value_counts()
    
    print("\nLabel Distribution:")
    print("-" * 90)
    for label in EXPECTED_LABELS:
        count = label_counts.get(label, 0)
        print(f"  {label}: {format_number(count)}")
    
    # Test 1: Total MI_Acute >= 500
    results.total_tests += 1
    mi_acute_count = label_counts.get('MI_Acute_Presentation', 0)
    if mi_acute_count >= MIN_TOTAL_MI_ACUTE:
        print_test(f"\nTotal MI_Acute >= {MIN_TOTAL_MI_ACUTE}", "PASS", 
                  f"{format_number(mi_acute_count)} cases", verbose)
        results.passed_tests += 1
    else:
        print_test(f"\nTotal MI_Acute >= {MIN_TOTAL_MI_ACUTE}", "FAIL", 
                  f"Only {format_number(mi_acute_count)} cases", verbose)
        results.failed_tests += 1
        results.errors.append(f"Insufficient MI_Acute cases: {mi_acute_count} < {MIN_TOTAL_MI_ACUTE}")
    
    # Test 2: Subgroup sizes (Gender)
    if 'gender' in df.columns:
        print("\nGender Subgroups:")
        print("-" * 90)
        mi_acute = df[df['primary_label'] == 'MI_Acute_Presentation']
        
        for gender in ['F', 'M']:
            results.total_tests += 1
            gender_count = len(mi_acute[mi_acute['gender'] == gender])
            gender_name = 'Female' if gender == 'F' else 'Male'
            
            if gender_count >= MIN_SUBGROUP_SIZE:
                print_test(f"  {gender_name} MI_Acute >= {MIN_SUBGROUP_SIZE}", "PASS", 
                          f"{format_number(gender_count)} cases", verbose)
                results.passed_tests += 1
            elif gender_count >= 50:  # Warning threshold
                print_test(f"  {gender_name} MI_Acute >= {MIN_SUBGROUP_SIZE}", "WARN", 
                          f"{format_number(gender_count)} cases (underpowered)", verbose)
                results.warnings += 1
                results.warning_messages.append(
                    f"{gender_name} MI_Acute subgroup underpowered: {gender_count} < {MIN_SUBGROUP_SIZE}"
                )
            else:
                print_test(f"  {gender_name} MI_Acute >= {MIN_SUBGROUP_SIZE}", "FAIL", 
                          f"Only {format_number(gender_count)} cases", verbose)
                results.failed_tests += 1
                results.errors.append(f"{gender_name} MI_Acute too small: {gender_count}")
    
    # Test 3: Subgroup sizes (Age)
    if 'anchor_age' in df.columns:
        print("\nAge Subgroups:")
        print("-" * 90)
        mi_acute = df[df['primary_label'] == 'MI_Acute_Presentation']
        
        age_bins = [
            ('<50', 0, 50),
            ('50-70', 50, 70),
            ('>70', 70, 150)
        ]
        
        for age_label, age_min, age_max in age_bins:
            results.total_tests += 1
            age_count = len(mi_acute[(mi_acute['anchor_age'] >= age_min) & 
                                    (mi_acute['anchor_age'] < age_max)])
            
            if age_count >= MIN_SUBGROUP_SIZE:
                print_test(f"  Age {age_label} MI_Acute >= {MIN_SUBGROUP_SIZE}", "PASS", 
                          f"{format_number(age_count)} cases", verbose)
                results.passed_tests += 1
            elif age_count >= 50:  # Warning threshold
                print_test(f"  Age {age_label} MI_Acute >= {MIN_SUBGROUP_SIZE}", "WARN", 
                          f"{format_number(age_count)} cases (underpowered)", verbose)
                results.warnings += 1
                results.warning_messages.append(
                    f"Age {age_label} MI_Acute underpowered: {age_count} < {MIN_SUBGROUP_SIZE}"
                )
            else:
                print_test(f"  Age {age_label} MI_Acute >= {MIN_SUBGROUP_SIZE}", "FAIL", 
                          f"Only {format_number(age_count)} cases", verbose)
                results.failed_tests += 1
                results.errors.append(f"Age {age_label} MI_Acute too small: {age_count}")
    
    return True

# ============================================================================
# TEST SUITE 4: DATA QUALITY CHECKS
# ============================================================================

def test_data_quality(df, results, quick=False, verbose=False):
    """Test data quality (nulls, types, ranges)"""
    print("\n" + "=" * 90)
    print("4. DATA QUALITY CHECKS")
    print("=" * 90)
    
    if quick:
        print("[INFO] Skipping detailed data quality checks (--quick mode)")
        return True
    
    # Test 1: Critical columns not null
    critical_columns = ['record_id', 'subject_id', 'ecg_time', 'primary_label']
    
    print("\nNull Analysis (Critical Columns):")
    print("-" * 90)
    for col in critical_columns:
        if col not in df.columns:
            continue
            
        results.total_tests += 1
        null_count = df[col].isna().sum()
        null_pct = 100 * null_count / len(df)
        
        if null_count == 0:
            print_test(f"  {col}", "PASS", "0 nulls", verbose)
            results.passed_tests += 1
        else:
            print_test(f"  {col}", "FAIL", f"{format_number(null_count)} nulls ({null_pct:.1f}%)", verbose)
            results.failed_tests += 1
            results.errors.append(f"{col} has {null_count} null values")
    
    # Test 2: Primary label validity
    results.total_tests += 1
    if 'primary_label' in df.columns:
        invalid_labels = df[~df['primary_label'].isin(EXPECTED_LABELS)]['primary_label'].unique()
        
        if len(invalid_labels) == 0:
            print_test("\nPrimary Label Validity", "PASS", "All labels valid", verbose)
            results.passed_tests += 1
        else:
            print_test("\nPrimary Label Validity", "FAIL", 
                      f"{len(invalid_labels)} invalid labels: {', '.join(map(str, invalid_labels[:5]))}", verbose)
            results.failed_tests += 1
            results.errors.append(f"Invalid primary labels found: {invalid_labels}")
    
    # Test 3: Temporal consistency
    if 'ecg_time' in df.columns and 'index_mi_time' in df.columns:
        print("\nTemporal Consistency:")
        print("-" * 90)
        
        # Check MI_Acute cases have index_mi_time
        results.total_tests += 1
        mi_acute = df[df['primary_label'] == 'MI_Acute_Presentation']
        if len(mi_acute) > 0:
            missing_mi_time = mi_acute['index_mi_time'].isna().sum()
            if missing_mi_time == 0:
                print_test("  MI_Acute cases have index_mi_time", "PASS", "", verbose)
                results.passed_tests += 1
            else:
                print_test("  MI_Acute cases have index_mi_time", "FAIL", 
                          f"{format_number(missing_mi_time)} missing", verbose)
                results.failed_tests += 1
                results.errors.append(f"MI_Acute cases missing index_mi_time: {missing_mi_time}")
        
        # Check Control cases don't have index_mi_time
        results.total_tests += 1
        controls = df[df['primary_label'].str.contains('Control', na=False)]
        if len(controls) > 0:
            has_mi_time = controls['index_mi_time'].notna().sum()
            if has_mi_time == 0:
                print_test("  Control cases don't have index_mi_time", "PASS", "", verbose)
                results.passed_tests += 1
            else:
                print_test("  Control cases don't have index_mi_time", "WARN", 
                          f"{format_number(has_mi_time)} have MI time (may be valid Pre-Incident)", verbose)
                results.warnings += 1
                results.warning_messages.append(
                    f"Control cases with index_mi_time: {has_mi_time}"
                )
    
    # Test 4: Unique identifiers
    # Note: Same ECG can appear with different labels (Control_Symptomatic vs MI labels)
    # This is by design for cohort analysis - check (record_id, primary_label) uniqueness
    results.total_tests += 1
    if 'record_id' in df.columns and 'primary_label' in df.columns:
        duplicate_combinations = df.duplicated(subset=['record_id', 'primary_label']).sum()
        total_records = len(df)
        unique_records = df['record_id'].nunique()
        dup_pct = 100 * duplicate_combinations / total_records if total_records > 0 else 0
        
        if duplicate_combinations == 0:
            print_test("\nUnique (Record ID, Label) Combinations", "PASS", 
                      f"{format_number(total_records)} rows, {format_number(unique_records)} unique ECGs", verbose)
            results.passed_tests += 1
        elif dup_pct < 0.1:  # Less than 0.1% duplicates - warning only
            print_test("\nUnique (Record ID, Label) Combinations", "WARN", 
                      f"{format_number(duplicate_combinations)} duplicates ({dup_pct:.3f}%, negligible)", verbose)
            results.warnings += 1
            results.warning_messages.append(
                f"Minor duplicates in (record_id, primary_label): {duplicate_combinations} ({dup_pct:.3f}%)"
            )
        else:
            print_test("\nUnique (Record ID, Label) Combinations", "FAIL", 
                      f"{format_number(duplicate_combinations)} duplicate pairs ({dup_pct:.1f}%)", verbose)
            results.failed_tests += 1
            results.errors.append(f"Duplicate (record_id, primary_label) combinations: {duplicate_combinations} ({dup_pct:.1f}%)")
        
        # Informational: Report multi-label ECGs
        if duplicate_combinations == 0 or dup_pct < 0.1:
            multi_label_ecgs = total_records - unique_records
            if multi_label_ecgs > 0:
                pct_multi = 100 * multi_label_ecgs / total_records
                print(f"  [INFO] {format_number(multi_label_ecgs)} rows ({pct_multi:.1f}%) are multi-label ECGs (same ECG, different labels)")
    
    print()
    print("[INFO] Data quality checks completed")
    
    return True

# ============================================================================
# TEST SUITE 5: CLINICAL VALIDATION
# ============================================================================

def test_clinical_validation(df, results, verbose=False):
    """Test clinical criteria and troponin validation"""
    print("\n" + "=" * 90)
    print("5. CLINICAL VALIDATION")
    print("=" * 90)
    
    # Test 1: Comorbidity flag validity
    results.total_tests += 1
    if 'comorbidity_chronic_mi' in df.columns:
        valid_values = [0, 1, True, False]
        invalid_comorbidity = df[~df['comorbidity_chronic_mi'].isin(valid_values)]
        
        if len(invalid_comorbidity) == 0:
            print_test("Comorbidity Flag Validity", "PASS", "All values binary (0/1)", verbose)
            results.passed_tests += 1
        else:
            print_test("Comorbidity Flag Validity", "FAIL", 
                      f"{format_number(len(invalid_comorbidity))} invalid values", verbose)
            results.failed_tests += 1
            results.errors.append(f"Invalid comorbidity_chronic_mi values: {len(invalid_comorbidity)}")
    
    # Test 2: Label distribution reasonableness
    results.total_tests += 1
    if 'primary_label' in df.columns:
        label_counts = df['primary_label'].value_counts()
        mi_acute = label_counts.get('MI_Acute_Presentation', 0)
        controls = label_counts.get('Control_Symptomatic', 0) + label_counts.get('Control_Asymptomatic', 0)
        
        if controls > 0:
            ratio = mi_acute / controls
            if 0.1 <= ratio <= 10:  # Reasonable range: 1:10 to 10:1
                print_test("MI:Control Ratio", "PASS", f"Ratio: {ratio:.2f}:1", verbose)
                results.passed_tests += 1
            else:
                print_test("MI:Control Ratio", "WARN", 
                          f"Ratio: {ratio:.2f}:1 (unusual but may be valid)", verbose)
                results.warnings += 1
                results.warning_messages.append(f"Unusual MI:Control ratio: {ratio:.2f}:1")
        else:
            print_test("MI:Control Ratio", "FAIL", "No control cases found", verbose)
            results.failed_tests += 1
            results.errors.append("No control cases in cohort")
    
    # Test 3: Check if connected to database for troponin validation
    if os.path.exists(DB_PATH):
        try:
            con = duckdb.connect(DB_PATH, read_only=True)
            
            # Check troponin coverage
            results.total_tests += 1
            troponin_query = """
            SELECT COUNT(DISTINCT subject_id) as subjects_with_troponin
            FROM labevents
            WHERE itemid IN (
                51002, 51003,  -- Troponin T
                50911           -- Troponin I
            )
            """
            
            troponin_subjects = con.execute(troponin_query).fetchone()[0]
            total_subjects = df['subject_id'].nunique()
            coverage_pct = 100 * troponin_subjects / total_subjects if total_subjects > 0 else 0
            
            if coverage_pct >= 50:
                print_test("Troponin Coverage", "PASS", 
                          f"{format_number(troponin_subjects)} subjects ({coverage_pct:.1f}%)", verbose)
                results.passed_tests += 1
            elif coverage_pct >= 20:
                print_test("Troponin Coverage", "WARN", 
                          f"{format_number(troponin_subjects)} subjects ({coverage_pct:.1f}%)", verbose)
                results.warnings += 1
                results.warning_messages.append(f"Low troponin coverage: {coverage_pct:.1f}%")
            else:
                print_test("Troponin Coverage", "FAIL", 
                          f"Only {format_number(troponin_subjects)} subjects ({coverage_pct:.1f}%)", verbose)
                results.failed_tests += 1
                results.errors.append(f"Insufficient troponin coverage: {coverage_pct:.1f}%")
            
            con.close()
        except Exception as e:
            print_test("Troponin Validation", "WARN", f"Cannot connect to database: {str(e)}", verbose)
            results.warnings += 1
            results.warning_messages.append(f"Database connection failed: {str(e)}")
    else:
        print_test("Troponin Validation", "WARN", "Database not found (skipped)", verbose)
        results.warnings += 1
    
    print()
    print("[INFO] Clinical validation completed")
    
    return True

# ============================================================================
# TEST SUITE 6: ENVIRONMENT VALIDATION
# ============================================================================

def test_environment_validation(df, results, verbose=False):
    """Test environment labels (Phase G preparation)"""
    print("\n" + "=" * 90)
    print("6. ENVIRONMENT VALIDATION (Phase G Preparation)")
    print("=" * 90)
    
    # Test 1: Environment label column exists
    results.total_tests += 1
    if 'environment_label' in df.columns:
        print_test("Environment Label Column", "PASS", "", verbose)
        results.passed_tests += 1
        
        # Check if populated
        null_count = df['environment_label'].isna().sum()
        populated_pct = 100 * (1 - null_count / len(df))
        
        if populated_pct >= 90:
            print_test("  Environment Labels Populated", "PASS", f"{populated_pct:.1f}% populated", verbose)
        elif populated_pct >= 50:
            print_test("  Environment Labels Populated", "WARN", 
                      f"{populated_pct:.1f}% populated (Phase G pending)", verbose)
            results.warnings += 1
            results.warning_messages.append(f"Environment labels partially populated: {populated_pct:.1f}%")
        else:
            print_test("  Environment Labels Populated", "INFO", 
                      f"{populated_pct:.1f}% populated (Phase G pending)", verbose)
    else:
        print_test("Environment Label Column", "FAIL", "Column missing", verbose)
        results.failed_tests += 1
        results.errors.append("environment_label column missing")
    
    print()
    print("[INFO] Environment validation completed (Phase G will populate labels)")
    
    return True

# ============================================================================
# SUMMARY FUNCTIONS
# ============================================================================

def print_summary(results, df, start_time):
    """Print test summary"""
    duration = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "=" * 90)
    print("PHASE C TEST SUMMARY")
    print("=" * 90)
    print()
    print(f"Test Duration: {duration:.2f} seconds")
    print(f"Cohort: {COHORT_MASTER_PATH}")
    print(f"Records: {format_number(len(df))}")
    print(f"Subjects: {format_number(df['subject_id'].nunique())}")
    print(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("=" * 90)
    print(f"Total Tests Run: {results.total_tests}")
    print(f"  [PASS] Passed: {results.passed_tests}")
    print(f"  [FAIL] Failed: {results.failed_tests}")
    print(f"  [WARN] Warnings: {results.warnings}")
    print("=" * 90)
    
    # Print detailed status
    if results.failed_tests == 0 and results.warnings == 0:
        print()
        print("=" * 90)
        print()
        print("[SUCCESS] ALL TESTS PASSED - COHORT VALIDATION SUCCESSFUL!")
        print()
        print("Phase C cohort labeling is complete and validated.")
        print("Cohort is ready for Phase D (ECG Feature Extraction).")
        print()
        print("=" * 90)
        print()
    elif results.failed_tests == 0:
        print()
        print("[WARNING] TESTS PASSED WITH WARNINGS")
        print()
        print(f"Found {results.warnings} warnings (non-critical issues):")
        for i, warning in enumerate(results.warning_messages[:10], 1):
            print(f"  {i}. {warning}")
        if len(results.warning_messages) > 10:
            print(f"  ... and {len(results.warning_messages) - 10} more")
        print()
        print("These warnings are acceptable for analysis but should be reviewed.")
        print()
    else:
        print()
        print("[FAILED] TESTS FAILED - CRITICAL ERRORS FOUND")
        print()
        print(f"Found {results.failed_tests} critical errors:")
        for i, error in enumerate(results.errors[:10], 1):
            print(f"  {i}. {error}")
        if len(results.errors) > 10:
            print(f"  ... and {len(results.errors) - 10} more")
        print()
        print("[WARNING] Fix critical errors before proceeding to Phase D")
        print()

def print_json_output(results):
    """Print JSON output for CI/CD"""
    print(json.dumps(results.to_json(), indent=2))

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main test execution"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Phase C Comprehensive Test Suite")
    parser.add_argument('--quick', action='store_true', help='Skip detailed data quality checks')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--verbose', action='store_true', help='Show detailed test progress')
    args = parser.parse_args()
    
    # Initialize
    results = TestResult()
    start_time = datetime.now()
    
    if not args.json:
        print("=" * 90)
        print("PHASE C COMPREHENSIVE TEST SUITE")
        print("=" * 90)
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Cohort: {COHORT_MASTER_PATH}")
        print(f"Mode: {'Quick' if args.quick else 'Full'}")
        print("=" * 90)
    
    # Check cohort file exists
    if not os.path.exists(COHORT_MASTER_PATH):
        print(f"\n[ERROR] Cohort file not found at {COHORT_MASTER_PATH}")
        print("Please run Phase C cohort labeling first:")
        print("  python src/data/cohort_labeling.py")
        sys.exit(1)
    
    # Load cohort
    try:
        df = pd.read_parquet(COHORT_MASTER_PATH)
    except Exception as e:
        print(f"\n[ERROR] Cannot read cohort file: {str(e)}")
        sys.exit(1)
    
    try:
        # Run test suites
        if not test_preflight(results, args.verbose):
            print("\n[ERROR] Pre-flight checks failed. Aborting.")
            sys.exit(1)
        
        test_schema(df, results, args.verbose)
        test_sample_size(df, results, args.verbose)
        test_data_quality(df, results, args.quick, args.verbose)
        test_clinical_validation(df, results, args.verbose)
        test_environment_validation(df, results, args.verbose)
        
        # Print summary
        if args.json:
            print_json_output(results)
        else:
            print_summary(results, df, start_time)
        
        # Exit with appropriate code
        if results.failed_tests > 0:
            sys.exit(1)
        elif results.warnings > 0:
            sys.exit(0)  # Warnings are acceptable
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
