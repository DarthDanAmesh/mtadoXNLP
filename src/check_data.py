# src/check_data.py
import pandas as pd
from pathlib import Path
import configparser

def check_data_files():
    # Read configuration from project root
    config = configparser.ConfigParser()
    project_root = Path(__file__).resolve().parent.parent  # cybersecurity_absa/
    config_path = project_root / 'config.ini'
    config.read(config_path)
    
    # Get directories relative to project root
    raw_data_dir = project_root / config['paths']['raw_data_dir']
    processed_data_dir = project_root / config['paths']['processed_data_dir']
    
    print("Checking data files...")
    print("=" * 60)
    
    # Check raw data files
    print("RAW DATA FILES:")
    print("-" * 30)
    raw_files_to_check = [
        'eurepoc_data_2025-09-27T19_04.xlsx',
        # Add other raw files if they exist
    ]
    
    for filename in raw_files_to_check:
        file_path = raw_data_dir / filename
        check_file(file_path, "Raw")
    
    # Check processed data files
    print("\nPROCESSED DATA FILES:")
    print("-" * 30)
    processed_files_to_check = [
        'eurepoc_processed.csv',
        'cisa_trafilatura_processed.csv', 
        'csis_trafilatura_processed.csv',
        'combined_dataset_phase1.csv',
        'dataset_with_bertopics.csv',
        'baseline_aspect_extraction.csv'
    ]
    
    for filename in processed_files_to_check:
        file_path = processed_data_dir / filename
        check_file(file_path, "Processed")

def check_file(file_path, file_type):
    """Check individual file and print detailed information"""
    print(f"{file_type}: {file_path.name}")
    print(f"  Location: {file_path}")
    
    if file_path.exists():
        file_size = file_path.stat().st_size
        print(f"Exists: Yes")
        print(f"Size: {file_size:,} bytes")
        
        if file_size > 0:
            try:
                # Try different file formats
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                else:
                    print(f"Unknown file format: {file_path.suffix}")
                    return
                
                print(f"Records: {len(df):,}")
                print(f"Columns: {len(df.columns)}")
                
                if len(df) > 0:
                    # Show column names (truncated if too many)
                    if len(df.columns) > 10:
                        print(f"Columns (first 10): {list(df.columns[:10])}...")
                    else:
                        print(f"Columns: {list(df.columns)}")
                    
                    # Show data types summary
                    dtype_summary = df.dtypes.value_counts()
                    print(f"Data types: {dict(dtype_summary)}")
                    
                    # Show first row sample (key columns only)
                    sample_cols = df.columns[:5] if len(df.columns) > 5 else df.columns
                    sample_data = {col: str(df.iloc[0][col])[:50] + "..." if len(str(df.iloc[0][col])) > 50 else str(df.iloc[0][col]) 
                                 for col in sample_cols}
                    print(f"Sample row: {sample_data}")
                    
                    # Check for missing values
                    missing_total = df.isnull().sum().sum()
                    missing_percent = (missing_total / (len(df) * len(df.columns))) * 100
                    print(f"Missing values: {missing_total:,} ({missing_percent:.1f}%)")
                    
            except Exception as e:
                print(f"Error reading: {e}")
        else:
            print(f"Content: Empty file")
    else:
        print(f"Exists: No")
    
    print()

def check_directory_contents():
    """Check what files actually exist in the data directories"""
    config = configparser.ConfigParser()
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / 'config.ini'
    config.read(config_path)
    
    raw_data_dir = project_root / config['paths']['raw_data_dir']
    processed_data_dir = project_root / config['paths']['processed_data_dir']
    
    print("\n" + "=" * 60)
    print("DIRECTORY CONTENTS:")
    print("=" * 60)
    
    print(f"\nRaw data directory ({raw_data_dir}):")
    if raw_data_dir.exists():
        raw_files = list(raw_data_dir.glob('*'))
        for file in raw_files:
            size = file.stat().st_size
            print(f"{file.name} ({size:,} bytes)")
    else:
        print("Directory does not exist")
    
    print(f"\nProcessed data directory ({processed_data_dir}):")
    if processed_data_dir.exists():
        processed_files = list(processed_data_dir.glob('*'))
        for file in processed_files:
            size = file.stat().st_size
            print(f"{file.name} ({size:,} bytes)")
    else:
        print("Directory does not exist")

def main():
    try:
        check_data_files()
        check_directory_contents()
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("=" * 60)
        print("Run the data collection scripts in this order:")
        print("1. python src/collect_eurepoc.py")
        print("2. python src/collect_cisa_trafilatura.py")
        print("3. python src/collect_csis_trafilatura.py")
        print("4. python src/preprocess_data.py")
        print("5. python src/run_bertopic.py")
        print("6. python src/run_pyabsa_baseline.py")
        print("7. python src/phase1_report.py")
        
    except Exception as e:
        print(f"Error during data check: {e}")

if __name__ == "__main__":
    main()