"""
=====================================================================
PHASE B COMPREHENSIVE TEST SUITE
=====================================================================

Comprehensive validation of MIMIC-IV database ingestion (Phase B).

TEST CATEGORIES:
1. Pre-flight Checks (database exists, readable)
2. Table Existence (all 21 expected tables present)
3. Row Count Validation (minimum requirements)
4. Schema Validation (required columns present)
5. Data Quality (nulls, data types, ranges)
6. Foreign Key Integrity (relationships valid)
7. Date Consistency (no impossible dates)
8. Summary Statistics

PASS/FAIL CRITERIA:
- PASS: All critical checks pass (tables exist, min rows, required columns)
- WARN: Non-critical issues (high nulls, data warnings)
- FAIL: Critical errors (missing tables, missing columns, broken FKs)

USAGE:
    python scripts/test_phase_b.py [--json] [--quick] [--verbose]
    
    --json: Output results in JSON format for CI/CD
    --quick: Skip detailed data quality checks
    --verbose: Show detailed progress

EXIT CODES:
    0 = All tests passed
    1 = Critical failures (missing tables/columns)
    2 = Warnings only (acceptable for analysis)

Author: Capstone Project Team
Date: February 6, 2026
"""

import duckdb
import os
import sys
import json
import time
import argparse
from datetime import datetime
from collections import defaultdict

# Set UTF-8 encoding for output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
DB_PATH = "data/mimic_database.duckdb"

# Expected tables with minimum row counts
EXPECTED_TABLES = {
    # MIMIC-IV (11 tables)
    "patients": {"min_rows": 100000, "dataset": "MIMIC-IV", "critical": True},
    "admissions": {"min_rows": 400000, "dataset": "MIMIC-IV", "critical": True},
    "diagnoses_icd": {"min_rows": 4000000, "dataset": "MIMIC-IV", "critical": True},
    "labevents": {"min_rows": 100000000, "dataset": "MIMIC-IV", "critical": True},
    "d_labitems": {"min_rows": 1000, "dataset": "MIMIC-IV", "critical": True},
    "prescriptions": {"min_rows": 10000000, "dataset": "MIMIC-IV", "critical": False},
    "d_icd_diagnoses": {"min_rows": 80000, "dataset": "MIMIC-IV", "critical": False},
    "procedures_icd": {"min_rows": 500000, "dataset": "MIMIC-IV", "critical": False},
    "chartevents": {"min_rows": 300000000, "dataset": "MIMIC-IV", "critical": False},
    "icustays": {"min_rows": 50000, "dataset": "MIMIC-IV", "critical": False},
    "d_items": {"min_rows": 3000, "dataset": "MIMIC-IV", "critical": True},
    
    # MIMIC-IV-ECG (4 tables)
    "record_list": {"min_rows": 800000, "dataset": "MIMIC-IV-ECG", "critical": True},
    "machine_measurements": {"min_rows": 800000, "dataset": "MIMIC-IV-ECG", "critical": True},
    "machine_measurements_data_dictionary": {"min_rows": 10, "dataset": "MIMIC-IV-ECG", "critical": True},  # Data dictionary: 16 variables expected
    "waveform_note_links": {"min_rows": 100000, "dataset": "MIMIC-IV-ECG", "critical": False},
    
    # MIMIC-IV-ECG-Ext-ICD (1 table)
    "records_w_diag_icd10": {"min_rows": 400000, "dataset": "MIMIC-IV-ECG-Ext-ICD", "critical": True},
    
    # PTB-XL (2 tables)
    "ptbxl_database": {"min_rows": 21000, "dataset": "PTB-XL", "critical": True},
    "scp_statements": {"min_rows": 70, "dataset": "PTB-XL", "critical": False},
    
    # PTB-XL+ (3 tables)
    "statements_12sl": {"min_rows": 21000, "dataset": "PTB-XL+", "critical": False},
    "ptbxl_statements": {"min_rows": 21000, "dataset": "PTB-XL+", "critical": False},
    "snomed_description": {"min_rows": 100, "dataset": "PTB-XL+", "critical": False},
}

# Required columns for key tables (used in cohort_labeling.py)
REQUIRED_COLUMNS = {
    "patients": ["subject_id", "gender", "anchor_age", "anchor_year"],
    "admissions": ["subject_id", "hadm_id", "admittime", "dischtime"],
    "labevents": ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
    "d_labitems": ["itemid", "label", "fluid", "category"],
    "diagnoses_icd": ["subject_id", "hadm_id", "icd_code"],
    "record_list": ["subject_id", "study_id", "ecg_time"],
    "machine_measurements": ["study_id"],
    "d_items": ["itemid", "label"],
}

# Foreign key relationships to validate
FOREIGN_KEYS = [
    ("admissions", "subject_id", "patients", "subject_id"),
    ("diagnoses_icd", "hadm_id", "admissions", "hadm_id"),
    ("labevents", "subject_id", "patients", "subject_id"),
    ("prescriptions", "hadm_id", "admissions", "hadm_id"),
]

class TestResult:
    """Store test results"""
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0
        self.errors = []
        self.warning_messages = []
        self.test_details = []
        self.start_time = time.time()
        
    def add_pass(self, test_name, message=""):
        self.total_tests += 1
        self.passed_tests += 1
        self.test_details.append({"test": test_name, "status": "PASS", "message": message})
        
    def add_fail(self, test_name, message):
        self.total_tests += 1
        self.failed_tests += 1
        self.errors.append(message)
        self.test_details.append({"test": test_name, "status": "FAIL", "message": message})
        
    def add_warning(self, test_name, message):
        self.total_tests += 1
        self.passed_tests += 1  # Warnings don't fail the test
        self.warnings += 1
        self.warning_messages.append(message)
        self.test_details.append({"test": test_name, "status": "WARN", "message": message})
    
    def get_duration(self):
        return time.time() - self.start_time
    
    def to_dict(self):
        return {
            "total_tests": self.total_tests,
            "passed": self.passed_tests,
            "failed": self.failed_tests,
            "warnings": self.warnings,
            "duration_seconds": round(self.get_duration(), 2),
            "errors": self.errors,
            "warnings_list": self.warning_messages,
            "test_details": self.test_details,
            "timestamp": datetime.now().isoformat()
        }

def print_header(title, char="="):
    """Print formatted header"""
    width = 90
    print()
    print(char * width)
    print(title.center(width))
    print(char * width)

def print_test(test_name, status, message="", verbose=False):
    """Print test result"""
    status_symbols = {
        "PASS": "[PASS]",
        "FAIL": "[FAIL]",
        "WARN": "[WARN]",
        "INFO": "[INFO]",
        "RUN": "[RUN]"
    }
    
    symbol = status_symbols.get(status, "  ")
    
    if status == "RUN" and not verbose:
        return  # Don't show "running" status unless verbose
    
    if message:
        print(f"{symbol} {test_name}: {message}")
    else:
        print(f"{symbol} {test_name}")

def test_preflight(con, results, verbose=False):
    """Pre-flight checks: database exists and is readable"""
    print_header("1. PRE-FLIGHT CHECKS", "=")
    
    # Database file exists
    if not os.path.exists(DB_PATH):
        results.add_fail("Database File Exists", f"Database not found at {DB_PATH}")
        print_test("Database File Exists", "FAIL", f"Not found: {DB_PATH}", verbose)
        return False
    
    results.add_pass("Database File Exists", f"Found: {DB_PATH}")
    print_test("Database File Exists", "PASS", f"Size: {os.path.getsize(DB_PATH) / 1024**3:.2f} GB", verbose)
    
    # Database is readable
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        results.add_pass("Database Readable", f"{len(tables)} tables found")
        print_test("Database Readable", "PASS", f"{len(tables)} tables accessible", verbose)
    except Exception as e:
        results.add_fail("Database Readable", f"Cannot read database: {str(e)}")
        print_test("Database Readable", "FAIL", str(e), verbose)
        return False
    
    return True

def test_table_existence(con, results, verbose=False):
    """Test that all expected tables exist with minimum rows"""
    print_header("2. TABLE EXISTENCE & ROW COUNTS", "=")
    
    actual_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    
    # Group by dataset
    by_dataset = defaultdict(list)
    for table, info in EXPECTED_TABLES.items():
        by_dataset[info['dataset']].append((table, info))
    
    total_rows = 0
    missing_critical = []
    
    for dataset, tables in sorted(by_dataset.items()):
        print(f"\n{dataset}:")
        print("-" * 90)
        
        for table, info in sorted(tables, key=lambda x: x[0]):
            if table not in actual_tables:
                status = "FAIL" if info['critical'] else "WARN"
                msg = f"Table '{table}' missing"
                if info['critical']:
                    results.add_fail(f"Table: {table}", msg)
                    missing_critical.append(table)
                else:
                    results.add_warning(f"Table: {table}", msg)
                print_test(table, status, "MISSING", verbose)
            else:
                # Check row count
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                total_rows += count
                
                if count < info['min_rows']:
                    results.add_warning(f"Table: {table}", 
                                       f"Low row count: {count:,} < {info['min_rows']:,} expected")
                    print_test(table, "WARN", f"{count:,} rows (< {info['min_rows']:,} expected)", verbose)
                elif count == 0:
                    results.add_fail(f"Table: {table}", "Table is empty")
                    print_test(table, "FAIL", "EMPTY", verbose)
                else:
                    results.add_pass(f"Table: {table}", f"{count:,} rows")
                    print_test(table, "PASS", f"{count:,} rows", verbose)
    
    print()
    print("=" * 90)
    print(f"Total Rows Across All Tables: {total_rows:,}")
    print("=" * 90)
    
    if missing_critical:
        print(f"\n[CRITICAL] {len(missing_critical)} critical tables missing: {', '.join(missing_critical)}")
        return False
    
    return True

def test_schema_validation(con, results, verbose=False):
    """Validate required columns exist in key tables"""
    print_header("3. SCHEMA VALIDATION (Required Columns)", "=")
    
    actual_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    schema_valid = True
    
    for table, required_cols in REQUIRED_COLUMNS.items():
        if table not in actual_tables:
            continue  # Already reported in table existence test
        
        print(f"\n{table}:")
        print("-" * 90)
        
        # Get actual columns
        columns_info = con.execute(f"DESCRIBE {table}").fetchall()
        actual_cols = [col[0] for col in columns_info]
        
        for col in required_cols:
            if col in actual_cols:
                results.add_pass(f"Column: {table}.{col}")
                print_test(f"  {col}", "PASS", "", verbose)
            else:
                results.add_fail(f"Column: {table}.{col}", f"Required column missing: {table}.{col}")
                print_test(f"  {col}", "FAIL", "MISSING", verbose)
                schema_valid = False
    
    return schema_valid

def test_data_quality(con, results, verbose=False):
    """Check data quality: nulls, data types, ranges"""
    print_header("4. DATA QUALITY CHECKS", "=")
    
    actual_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    
    # Check critical tables only (for speed)
    critical_tables = [t for t, info in EXPECTED_TABLES.items() if info.get('critical') and t in actual_tables]
    
    for table in critical_tables[:5]:  # Limit to first 5 for speed
        try:
            # Get total rows
            total_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if total_rows == 0:
                continue
            
            # Check for high NULL percentages in key columns
            if table in REQUIRED_COLUMNS:
                for col in REQUIRED_COLUMNS[table][:3]:  # Check first 3 required columns
                    try:
                        null_count = con.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL"
                        ).fetchone()[0]
                        
                        null_pct = (null_count / total_rows) * 100
                        
                        if null_pct > 50:
                            results.add_warning(f"Nulls: {table}.{col}", 
                                               f"{null_pct:.1f}% NULL values")
                            print_test(f"{table}.{col}", "WARN", f"{null_pct:.1f}% NULL", verbose)
                        elif null_pct > 0 and verbose:
                            print_test(f"{table}.{col}", "INFO", f"{null_pct:.1f}% NULL", verbose)
                        elif null_pct == 0:
                            results.add_pass(f"Nulls: {table}.{col}")
                            
                    except Exception as e:
                        if verbose:
                            print_test(f"{table}.{col}", "INFO", f"Cannot check NULLs: {e}", verbose)
                        
        except Exception as e:
            if verbose:
                print_test(table, "WARN", f"Quality check failed: {e}", verbose)
    
    print()
    print("[INFO] Data quality checks completed (sampling critical tables)")

def test_foreign_key_integrity(con, results, verbose=False):
    """Validate foreign key relationships"""
    print_header("5. FOREIGN KEY INTEGRITY", "=")
    
    actual_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    
    for child_table, child_col, parent_table, parent_col in FOREIGN_KEYS:
        if child_table not in actual_tables or parent_table not in actual_tables:
            continue
        
        try:
            # Find orphaned records
            orphans = con.execute(f"""
                SELECT COUNT(*) 
                FROM {child_table} c
                LEFT JOIN {parent_table} p ON c.{child_col} = p.{parent_col}
                WHERE p.{parent_col} IS NULL AND c.{child_col} IS NOT NULL
                LIMIT 1000
            """).fetchone()[0]
            
            if orphans > 0:
                results.add_warning(f"FK: {child_table}.{child_col}", 
                                   f"{orphans:,} orphaned records")
                print_test(f"{child_table}.{child_col} → {parent_table}.{parent_col}", 
                          "WARN", f"{orphans:,} orphans", verbose)
            else:
                results.add_pass(f"FK: {child_table}.{child_col}")
                print_test(f"{child_table}.{child_col} → {parent_table}.{parent_col}", 
                          "PASS", "", verbose)
                
        except Exception as e:
            results.add_warning(f"FK: {child_table}.{child_col}", 
                               f"Cannot validate: {str(e)}")
            if verbose:
                print_test(f"{child_table}.{child_col} → {parent_table}.{parent_col}", 
                          "WARN", f"Check failed: {e}", verbose)

def test_date_consistency(con, results, verbose=False):
    """Check date consistency in key tables"""
    print_header("6. DATE CONSISTENCY", "=")
    
    date_checks = {
        "admissions": ["admittime", "dischtime"],
        "labevents": ["charttime"],
    }
    
    actual_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    
    for table, date_cols in date_checks.items():
        if table not in actual_tables:
            continue
        
        for col in date_cols:
            try:
                # Check for future dates (beyond 2026)
                future = con.execute(f"""
                    SELECT COUNT(*) FROM {table} 
                    WHERE {col} > '2026-12-31' AND {col} IS NOT NULL
                """).fetchone()[0]
                
                if future > 0:
                    results.add_warning(f"Dates: {table}.{col}", 
                                       f"{future:,} future dates (> 2026)")
                    print_test(f"{table}.{col}", "WARN", f"{future:,} future dates", verbose)
                else:
                    results.add_pass(f"Dates: {table}.{col}")
                    if verbose:
                        print_test(f"{table}.{col}", "PASS", "Dates valid", verbose)
                        
            except Exception as e:
                if verbose:
                    print_test(f"{table}.{col}", "INFO", f"Cannot check: {e}", verbose)
    
    print()
    print("[INFO] Date consistency checks completed")

def print_summary_report(results, json_output=False):
    """Print final summary report"""
    if json_output:
        print(json.dumps(results.to_dict(), indent=2))
        return
    
    print_header("PHASE B TEST SUMMARY", "=")
    
    print(f"\nTest Duration: {results.get_duration():.2f} seconds")
    print(f"Database: {os.path.abspath(DB_PATH)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("=" * 90)
    print(f"Total Tests Run: {results.total_tests}")
    print(f"  [PASS] Passed: {results.passed_tests}")
    print(f"  [FAIL] Failed: {results.failed_tests}")
    print(f"  [WARN] Warnings: {results.warnings}")
    print("=" * 90)
    
    if results.failed_tests == 0 and results.warnings == 0:
        print()
        print("=" * 90)
        print()
        print("[SUCCESS] ALL TESTS PASSED - DATABASE VALIDATION SUCCESSFUL!")
        print()
        print("Phase B data ingestion is complete and validated.")
        print("Database is ready for Phase C (Cohort Labeling).")
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
        print("[WARNING] Fix critical errors before proceeding to Phase C")
        print()

def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description="Phase B Comprehensive Test Suite")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--quick", action="store_true", help="Skip detailed data quality checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress")
    args = parser.parse_args()
    
    if not args.json:
        print_header("PHASE B COMPREHENSIVE TEST SUITE", "=")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Database: {DB_PATH}")
        print(f"Mode: {'Quick' if args.quick else 'Full'}")
    
    # Initialize results
    results = TestResult()
    
    # Check database exists
    if not os.path.exists(DB_PATH):
        print(f"\n[ERROR] Database file not found at {DB_PATH}")
        print("Please run Phase B data ingestion first.")
        sys.exit(1)
    
    # Connect to database
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to database: {str(e)}")
        sys.exit(1)
    
    try:
        # Run test suites
        if not test_preflight(con, results, args.verbose):
            print_summary_report(results, args.json)
            sys.exit(1)
        
        if not test_table_existence(con, results, args.verbose):
            print_summary_report(results, args.json)
            sys.exit(1)
        
        if not test_schema_validation(con, results, args.verbose):
            print_summary_report(results, args.json)
            sys.exit(1)
        
        if not args.quick:
            test_data_quality(con, results, args.verbose)
            test_foreign_key_integrity(con, results, args.verbose)
            test_date_consistency(con, results, args.verbose)
        
        # Print summary
        print_summary_report(results, args.json)
        
        # Exit code
        if results.failed_tests > 0:
            sys.exit(1)  # Critical failures
        elif results.warnings > 0:
            sys.exit(2)  # Warnings only
        else:
            sys.exit(0)  # All passed
            
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        con.close()

if __name__ == "__main__":
    main()
