"""
Phase B: MIMIC-IV and MIMIC-ECG Data Ingestion Script
======================================================

This script ingests all necessary MIMIC-IV and MIMIC-ECG data into a local DuckDB database.
It follows a methodical approach:
    1. Database setup and connection
    2. Load MIMIC-IV tables (Parquet format)
    3. Load MIMIC-ECG tables (CSV format)
    4. Create performance indices

Author: Data Engineering Team
Date: October 23, 2025
"""

import duckdb
import os


# =============================================================================
# PATH CONFIGURATION - EDIT THESE PATHS TO MATCH YOUR LOCAL SETUP
# =============================================================================

# Base path to data directory
DATA_BASE_PATH = "data/raw/"

# Path to MIMIC-IV data directory (contains hosp/ and icu/ subdirectories with CSV files)
MIMIC_IV_HOSP_PATH = f"{DATA_BASE_PATH}MIMIC-IV-2.2/hosp/"
MIMIC_IV_ICU_PATH = f"{DATA_BASE_PATH}MIMIC-IV-2.2/icu/"

# Path to MIMIC-ECG data directory (contains CSV files)
MIMIC_ECG_PATH = f"{DATA_BASE_PATH}MIMIC-IV-ECG-1.0/"

# Path to MIMIC-ECG Extended ICD data directory
MIMIC_ECG_EXT_ICD_PATH = f"{DATA_BASE_PATH}MIMIC-IV-ECG-Ext-ICD-1.0.1/"

# Path to PTB-XL data directory
PTB_XL_PATH = f"{DATA_BASE_PATH}PTB-XL-1.0.3/"

# Path to PTB-XL+ data directory
PTB_XL_PLUS_PATH = f"{DATA_BASE_PATH}PTB-XL+-1.0.1/"


# =============================================================================
# MAIN INGESTION FUNCTION
# =============================================================================

def main():
    """
    Main function to orchestrate the complete data ingestion pipeline.
    Connects to DuckDB, loads all required tables, and creates indices.
    """
    
    # Database file path
    db_path = "data/mimic_database.duckdb"
    
    print("=" * 70)
    print("PHASE B: MIMIC DATA INGESTION")
    print("=" * 70)
    print(f"Database file: {db_path}")
    print(f"MIMIC-IV Hospital path: {MIMIC_IV_HOSP_PATH}")
    print(f"MIMIC-IV ICU path: {MIMIC_IV_ICU_PATH}")
    print(f"MIMIC-ECG path: {MIMIC_ECG_PATH}")
    print(f"MIMIC-ECG-Ext-ICD path: {MIMIC_ECG_EXT_ICD_PATH}")
    print(f"PTB-XL path: {PTB_XL_PATH}")
    print(f"PTB-XL+ path: {PTB_XL_PLUS_PATH}")
    print("=" * 70)
    print()
    
    # Connect to DuckDB database
    con = duckdb.connect(db_path)
    
    try:
        # =====================================================================
        # B.2: LOAD MIMIC-IV TABLES (CSV Format)
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("B.2: LOADING MIMIC-IV TABLES")
        print("=" * 70 + "\n")
        
        # Load patients table (from hosp directory)
        print("Loading 'patients' table...")
        con.execute(f"""
            CREATE TABLE patients AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}patients.csv', header=True)
        """)
        print("✓ 'patients' table loaded successfully.\n")
        
        # Load admissions table (from hosp directory)
        print("Loading 'admissions' table...")
        con.execute(f"""
            CREATE TABLE admissions AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}admissions.csv', header=True)
        """)
        print("✓ 'admissions' table loaded successfully.\n")
        
        # Load diagnoses_icd table (from hosp directory)
        print("Loading 'diagnoses_icd' table...")
        con.execute(f"""
            CREATE TABLE diagnoses_icd AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}diagnoses_icd.csv', header=True)
        """)
        print("✓ 'diagnoses_icd' table loaded successfully.\n")
        
        # Load labevents table (from hosp directory) - This is a large file
        print("Loading 'labevents' table (this may take several minutes for large files)...")
        con.execute(f"""
            CREATE TABLE labevents AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}labevents.csv', header=True)
        """)
        print("✓ 'labevents' table loaded successfully.\n")
        
        # Load d_labitems table (from hosp directory)
        print("Loading 'd_labitems' table...")
        con.execute(f"""
            CREATE TABLE d_labitems AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}d_labitems.csv', header=True)
        """)
        print("✓ 'd_labitems' table loaded successfully.\n")
        
        # Load chartevents table (from icu directory) - This is a very large file
        print("Loading 'chartevents' table (this may take several minutes for large files)...")
        con.execute(f"""
            CREATE TABLE chartevents AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_ICU_PATH}chartevents.csv', header=True)
        """)
        print("✓ 'chartevents' table loaded successfully.\n")
        
        # Load prescriptions table (from hosp directory)
        print("Loading 'prescriptions' table...")
        con.execute(f"""
            CREATE TABLE prescriptions AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}prescriptions.csv', header=True)
        """)
        print("✓ 'prescriptions' table loaded successfully.\n")
        
        # Load d_icd_diagnoses table (from hosp directory)
        print("Loading 'd_icd_diagnoses' table...")
        con.execute(f"""
            CREATE TABLE d_icd_diagnoses AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}d_icd_diagnoses.csv', header=True)
        """)
        print("✓ 'd_icd_diagnoses' table loaded successfully.\n")
        
        # Load procedures_icd table (from hosp directory)
        print("Loading 'procedures_icd' table...")
        con.execute(f"""
            CREATE TABLE procedures_icd AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_HOSP_PATH}procedures_icd.csv', header=True)
        """)
        print("✓ 'procedures_icd' table loaded successfully.\n")
        
        # Load icustays table (from icu directory)
        print("Loading 'icustays' table...")
        con.execute(f"""
            CREATE TABLE icustays AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_ICU_PATH}icustays.csv', header=True)
        """)
        print("✓ 'icustays' table loaded successfully.\n")
        
        # Load d_items table (from icu directory)
        print("Loading 'd_items' table...")
        con.execute(f"""
            CREATE TABLE d_items AS 
            SELECT * FROM read_csv_auto('{MIMIC_IV_ICU_PATH}d_items.csv', header=True)
        """)
        print("✓ 'd_items' table loaded successfully.\n")
        
        print("=" * 70)
        print("MIMIC-IV TABLES LOADED: 11/11")
        print("=" * 70)
        
        # =====================================================================
        # B.3: LOAD MIMIC-ECG TABLES (CSV Format)
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("B.3: LOADING MIMIC-ECG TABLES")
        print("=" * 70 + "\n")
        
        # Load record_list table
        print("Loading 'record_list' table...")
        con.execute(f"""
            CREATE TABLE record_list AS 
            SELECT * FROM read_csv_auto('{MIMIC_ECG_PATH}record_list.csv', header=True)
        """)
        print("✓ 'record_list' table loaded successfully.\n")
        
        # Load machine_measurements table
        print("Loading 'machine_measurements' table...")
        con.execute(f"""
            CREATE TABLE machine_measurements AS 
            SELECT * FROM read_csv_auto('{MIMIC_ECG_PATH}machine_measurements.csv', header=True)
        """)
        print("✓ 'machine_measurements' table loaded successfully.\n")
        
        # Load machine_measurements_data_dictionary table
        print("Loading 'machine_measurements_data_dictionary' table...")
        con.execute(f"""
            CREATE TABLE machine_measurements_data_dictionary AS 
            SELECT * FROM read_csv_auto('{MIMIC_ECG_PATH}machine_measurements_data_dictionary.csv', header=True)
        """)
        print("✓ 'machine_measurements_data_dictionary' table loaded successfully.\n")
        
        # Load waveform_note_links table
        print("Loading 'waveform_note_links' table...")
        con.execute(f"""
            CREATE TABLE waveform_note_links AS 
            SELECT * FROM read_csv_auto('{MIMIC_ECG_PATH}waveform_note_links.csv', header=True)
        """)
        print("✓ 'waveform_note_links' table loaded successfully.\n")
        
        print("=" * 70)
        print("MIMIC-ECG TABLES LOADED: 4/4")
        print("=" * 70)
        
        # =====================================================================
        # B.3.1: LOAD MIMIC-ECG-EXT-ICD TABLES (CSV Format)
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("B.3.1: LOADING MIMIC-ECG-EXT-ICD TABLES")
        print("=" * 70 + "\n")
        
        # Load records_w_diag_icd10 table
        print("Loading 'records_w_diag_icd10' table...")
        con.execute(f"""
            CREATE TABLE records_w_diag_icd10 AS 
            SELECT * FROM read_csv_auto('{MIMIC_ECG_EXT_ICD_PATH}records_w_diag_icd10.csv', header=True)
        """)
        print("✓ 'records_w_diag_icd10' table loaded successfully.\n")
        
        print("=" * 70)
        print("MIMIC-ECG-EXT-ICD TABLES LOADED: 1/1")
        print("=" * 70)
        
        # =====================================================================
        # B.3.2: LOAD PTB-XL TABLES (CSV Format)
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("B.3.2: LOADING PTB-XL TABLES")
        print("=" * 70 + "\n")
        
        # Load ptbxl_database table
        print("Loading 'ptbxl_database' table...")
        con.execute(f"""
            CREATE TABLE ptbxl_database AS 
            SELECT * FROM read_csv_auto('{PTB_XL_PATH}ptbxl_database.csv', header=True)
        """)
        print("✓ 'ptbxl_database' table loaded successfully.\n")
        
        # Load scp_statements table
        print("Loading 'scp_statements' table...")
        con.execute(f"""
            CREATE TABLE scp_statements AS 
            SELECT * FROM read_csv_auto('{PTB_XL_PATH}scp_statements.csv', header=True)
        """)
        print("✓ 'scp_statements' table loaded successfully.\n")
        
        print("=" * 70)
        print("PTB-XL TABLES LOADED: 2/2")
        print("=" * 70)
        
        # =====================================================================
        # B.3.3: LOAD PTB-XL+ TABLES (CSV Format)
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("B.3.3: LOADING PTB-XL+ TABLES")
        print("=" * 70 + "\n")
        
        # Load 12sl_statements table
        print("Loading '12sl_statements' table...")
        con.execute(f"""
            CREATE TABLE statements_12sl AS 
            SELECT * FROM read_csv_auto('{PTB_XL_PLUS_PATH}labels/12sl_statements.csv', header=True)
        """)
        print("✓ '12sl_statements' table loaded successfully.\n")
        
        # Load ptbxl_statements table
        print("Loading 'ptbxl_statements' table...")
        con.execute(f"""
            CREATE TABLE ptbxl_statements AS 
            SELECT * FROM read_csv_auto('{PTB_XL_PLUS_PATH}labels/ptbxl_statements.csv', header=True)
        """)
        print("✓ 'ptbxl_statements' table loaded successfully.\n")
        
        # Load snomed_description table
        print("Loading 'snomed_description' table...")
        con.execute(f"""
            CREATE TABLE snomed_description AS 
            SELECT * FROM read_csv_auto('{PTB_XL_PLUS_PATH}labels/snomed_description.csv', header=True)
        """)
        print("✓ 'snomed_description' table loaded successfully.\n")
        
        print("=" * 70)
        print("PTB-XL+ TABLES LOADED: 3/3")
        print("=" * 70)
        
        # =====================================================================
        # B.4: CREATE INDICES FOR PERFORMANCE OPTIMIZATION
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("B.4: CREATING PERFORMANCE INDICES")
        print("=" * 70 + "\n")
        
        print("Creating index on labevents.subject_id...")
        con.execute("CREATE INDEX idx_labevents_subject ON labevents(subject_id)")
        print("✓ Index 'idx_labevents_subject' created.\n")
        
        print("Creating index on labevents.hadm_id...")
        con.execute("CREATE INDEX idx_labevents_hadm ON labevents(hadm_id)")
        print("✓ Index 'idx_labevents_hadm' created.\n")
        
        print("Creating index on labevents.charttime...")
        con.execute("CREATE INDEX idx_labevents_charttime ON labevents(charttime)")
        print("✓ Index 'idx_labevents_charttime' created.\n")
        
        print("Creating index on chartevents.subject_id...")
        con.execute("CREATE INDEX idx_chartevents_subject ON chartevents(subject_id)")
        print("✓ Index 'idx_chartevents_subject' created.\n")
        
        print("Creating index on chartevents.charttime...")
        con.execute("CREATE INDEX idx_chartevents_charttime ON chartevents(charttime)")
        print("✓ Index 'idx_chartevents_charttime' created.\n")
        
        print("Creating index on diagnoses_icd.hadm_id...")
        con.execute("CREATE INDEX idx_diagnoses_hadm ON diagnoses_icd(hadm_id)")
        print("✓ Index 'idx_diagnoses_hadm' created.\n")
        
        print("=" * 70)
        print("INDICES CREATED: 6/6")
        print("=" * 70)
        
        # =====================================================================
        # COMPLETION MESSAGE
        # =====================================================================
        
        print("\n" + "=" * 70)
        print("Phase B ingestion complete. Database is ready.")
        print("=" * 70)
        print(f"\nDatabase location: {os.path.abspath(db_path)}")
        print("Total tables loaded: 21")
        print("  - MIMIC-IV: 11 tables")
        print("  - MIMIC-ECG: 4 tables")
        print("  - MIMIC-ECG-Ext-ICD: 1 table")
        print("  - PTB-XL: 2 tables")
        print("  - PTB-XL+: 3 tables")
        print("Total indices created: 6")
        print("\nYou can now proceed to Phase C: Data Processing and Analysis")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR DURING INGESTION")
        print("=" * 70)
        print(f"An error occurred: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nPlease check:")
        print("  1. File paths are correct")
        print("  2. All required files exist")
        print("  3. Files are not corrupted")
        print("  4. You have sufficient disk space")
        print("=" * 70 + "\n")
        raise
        
    finally:
        # Always close the database connection
        con.close()
        print("Database connection closed.")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
