import pandas as pd
from pathlib import Path
import configparser

def check_data_files():
    """Check if raw and processed data files exist and provide details."""
    # Read configuration from project root
    config = configparser.ConfigParser()
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.ini'
    print(f"Reading config from: {config_path}")

    config.read(config_path)
    
    # Debugging: Print config sections
    print(f"Config sections: {config.sections()}")
    
    if 'paths' not in config:
        print("Error: 'paths' section not found in config.ini")
        return

    # Get directories relative to project root
    try:
        raw_data_dir = project_root / config['paths']['raw_data_dir']
        processed_data_dir = project_root / config['paths']['processed_data_dir']
    except KeyError as e:
        print(f"Error: Missing {e} key in config.ini 'paths' section.")
        return

    print("Checking data files...")
    print("=" * 60)
    
    # --- Check RAW data files (all .xlsx, .xls) ---
    print("RAW DATA FILES:")
    print("-" * 30)
    raw_files = list(raw_data_dir.glob("*.xlsx")) + list(raw_data_dir.glob("*.xls"))
    if not raw_files:
        print("  No Excel files (.xlsx/.xls) found in raw data directory.")
    else:
        for file_path in sorted(raw_files):
            check_file(file_path, "Raw")

    # --- Check PROCESSED data files (all .csv) ---
    print("\nPROCESSED DATA FILES:")
    print("-" * 30)
    processed_files = list(processed_data_dir.glob("*.csv"))
    if not processed_files:
        print("  No CSV files found in processed data directory.")
    else:
        for file_path in sorted(processed_files):
            check_file(file_path, "Processed")


def check_file(file_path, file_type):
    """Check individual file and print detailed information."""
    print(f"{file_type}: {file_path.name}")
    print(f"  Location: {file_path}")
    
    if file_path.exists():
        try:
            file_size = file_path.stat().st_size
            print(f"Exists: Yes")
            print(f"Size: {file_size:,} bytes")
            
            if file_size == 0:
                print("Content: Empty file")
            else:
                # Read file based on extension
                suffix = file_path.suffix.lower()
                if suffix == '.csv':
                    df = pd.read_csv(file_path)
                elif suffix in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                else:
                    print(f"Unsupported format: {suffix}")
                    print()
                    return

                print(f"Records: {len(df):,}")
                print(f"Columns: {len(df.columns)}")
                
                if len(df) > 0:
                    # Show column names (truncated if too many)
                    cols_to_show = df.columns[:10]
                    if len(df.columns) > 10:
                        print(f"Columns (first 10): {list(cols_to_show)}...")
                    else:
                        print(f"Columns: {list(df.columns)}")
                    
                    # Data types summary
                    dtype_summary = df.dtypes.value_counts()
                    print(f"Data types: {dict(dtype_summary)}")
                    
                    # Sample first row (first 5 columns)
                    sample_cols = df.columns[:5]
                    sample_data = {}
                    for col in sample_cols:
                        val = df.iloc[0][col]
                        val_str = str(val)
                        if len(val_str) > 50:
                            val_str = val_str[:50] + "..."
                        sample_data[col] = val_str
                    print(f"Sample row: {sample_data}")
                    
                    # Missing values
                    missing_total = df.isnull().sum().sum()
                    total_cells = len(df) * len(df.columns)
                    missing_percent = (missing_total / total_cells) * 100 if total_cells > 0 else 0
                    print(f"Missing values: {missing_total:,} ({missing_percent:.1f}%)")
                    
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print("Exists: No")
    
    print()


def check_directory_contents():
    """List all files in raw and processed directories (not just .csv/.xlsx)."""
    config = configparser.ConfigParser()
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / 'config.ini'
    config.read(config_path)
    
    raw_data_dir = project_root / config['paths']['raw_data_dir']
    processed_data_dir = project_root / config['paths']['processed_data_dir']
    
    print("\n" + "=" * 60)
    print("DIRECTORY CONTENTS (ALL FILES):")
    print("=" * 60)
    
    print(f"\nRaw data directory ({raw_data_dir}):")
    if raw_data_dir.exists():
        files = sorted(raw_data_dir.iterdir())
        if files:
            for f in files:
                size = f.stat().st_size
                print(f"  {f.name} ({size:,} bytes)")
        else:
            print("  [Empty]")
    else:
        print("  Directory does not exist")
    
    print(f"\nProcessed data directory ({processed_data_dir}):")
    if processed_data_dir.exists():
        files = sorted(processed_data_dir.iterdir())
        if files:
            for f in files:
                size = f.stat().st_size
                print(f"  {f.name} ({size:,} bytes)")
        else:
            print("  [Empty]")
    else:
        print("  Directory does not exist")


def main():
    """Main function to check data files and directory contents."""
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
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()