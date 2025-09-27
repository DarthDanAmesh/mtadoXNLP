# src/collect_eurepoc.py
import pandas as pd
import configparser
from pathlib import Path

class EuRepoCDataCollector:
    def __init__(self):
        # Read configuration from project root
        self.config = configparser.ConfigParser()
        self.project_root = Path(__file__).resolve().parent.parent  # cybersecurity_absa/
        config_path = self.project_root / 'config.ini'
        self.config.read(config_path)

    def load_local_dataset(self, file_path):
        """Load EuRepoC dataset from local CSV or Excel file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()
            
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                print(f"Error: Unsupported file format {file_path.suffix}")
                return pd.DataFrame()
                
            print(f"Successfully loaded EuRepoC dataset with {len(df)} records")
            print(f"Dataset columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return pd.DataFrame()

    def process_incidents(self, df):
        """Process and clean EuRepoC data based on the actual dataset structure"""
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Map the actual EuRepoC columns to our standard format
        column_mapping = {
            'id': 'ID',
            'title': 'name',
            'description': 'description', 
            'date': 'start_date',
            'incident_type': 'incident_type',
            'severity': 'unweighted_cyber_intensity',
            'receiver_country': 'receiver_country',
            'receiver_category': 'receiver_category',
            'initiator_country': 'initiator_country', 
            'initiator_category': 'initiator_category',
            'cyber_intensity': 'weighted_cyber_intensity',
            'impact_indicator': 'impact_indicator',
            'mitre_techniques': 'MITRE_initial_access',
            'data_theft': 'data_theft',
            'disruption': 'disruption',
            'sources_url': 'sources_url'
        }
        
        # Rename columns to standard names
        for new_col, old_col in column_mapping.items():
            if old_col in df_processed.columns:
                df_processed[new_col] = df_processed[old_col]
        
        # Ensure essential columns exist
        essential_columns = ['title', 'description']
        for col in essential_columns:
            if col not in df_processed.columns:
                print(f"Warning: Essential column '{col}' not found in dataset")
                if col == 'title' and 'name' in df_processed.columns:
                    df_processed['title'] = df_processed['name']
                elif col == 'description':
                    df_processed['description'] = 'No description available'
        
        # Clean and standardize data
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
        
        # Fill missing text values
        text_columns = ['title', 'description', 'incident_type', 'receiver_country', 
                       'initiator_country', 'receiver_category', 'initiator_category']
        for col in text_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        # Convert boolean columns
        bool_columns = ['data_theft', 'disruption']
        for col in bool_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(False).astype(bool)
        
        # Convert numeric columns
        numeric_columns = ['severity', 'cyber_intensity', 'impact_indicator']
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
        
        # Create a combined impact description
        if all(col in df_processed.columns for col in ['data_theft', 'disruption', 'cyber_intensity']):
            impact_parts = []
            if df_processed['data_theft'].any():
                impact_parts.append('Data Theft')
            if df_processed['disruption'].any():
                impact_parts.append('Service Disruption')
            if 'impact_indicator' in df_processed.columns:
                impact_parts.append(f"Impact Level: {df_processed['impact_indicator'].mean():.1f}")
            
            df_processed['impact_description'] = ' | '.join(impact_parts) if impact_parts else 'No significant impact'
        
        # Add source identifier
        df_processed['source'] = 'EuRepoC'
        
        # Select final columns for analysis
        final_columns = [
            'id', 'title', 'description', 'date', 'incident_type', 'severity',
            'receiver_country', 'receiver_category', 'initiator_country', 'initiator_category',
            'cyber_intensity', 'impact_indicator', 'data_theft', 'disruption',
            'mitre_techniques', 'impact_description', 'sources_url', 'source'
        ]
        
        # Keep only columns that exist in the dataframe
        available_columns = [col for col in final_columns if col in df_processed.columns]
        
        return df_processed[available_columns]

    def analyze_dataset(self, df):
        """Provide basic analysis of the EuRepoC dataset"""
        if df.empty:
            print("No data to analyze")
            return
            
        print("\n=== EuRepoC Dataset Analysis ===")
        print(f"Total incidents: {len(df)}")
        
        if 'date' in df.columns:
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        if 'incident_type' in df.columns:
            incident_types = df['incident_type'].value_counts()
            print(f"Top 5 incident types:")
            for incident_type, count in incident_types.head(5).items():
                print(f"  - {incident_type}: {count} incidents")
        
        if 'receiver_country' in df.columns:
            target_countries = df['receiver_country'].value_counts()
            print(f"Top 5 target countries:")
            for country, count in target_countries.head(5).items():
                print(f"  - {country}: {count} incidents")
        
        if 'initiator_country' in df.columns:
            initiator_countries = df['initiator_country'].value_counts()
            print(f"Top 5 initiator countries:")
            for country, count in initiator_countries.head(5).items():
                print(f"  - {country}: {count} incidents")
        
        if 'severity' in df.columns:
            print(f"Average severity: {df['severity'].mean():.2f}")
            print(f"Severity range: {df['severity'].min()} - {df['severity'].max()}")
        
        print("=" * 40)

def main():
    # Initialize collector
    eurepoc_collector = EuRepoCDataCollector()
    
    # Get paths from config relative to project root
    raw_data_dir = eurepoc_collector.project_root / eurepoc_collector.config['paths']['raw_data_dir']
    processed_data_dir = eurepoc_collector.project_root / eurepoc_collector.config['paths']['processed_data_dir']
    
    # Ensure directories exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for EuRepoC dataset files in the raw data directory
    eurepoc_files = sorted(raw_data_dir.glob('*[eE]u[Rr]epo[Cc]*'), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not eurepoc_files:
        print("No EuRepoC dataset files found in raw data directory.")
        print(f"Expected location: {raw_data_dir}")
        print("Please download the EuRepoC dataset and place it in the data/raw/ directory.")
        print("Expected filename pattern: *eurepoc* or *EuRepoC* (CSV or Excel format)")
        return
    
    # Process each found file
    for file_path in eurepoc_files:
        print(f"Processing EuRepoC dataset: {file_path.name}")
        
        # Load the dataset
        eurepoc_raw = eurepoc_collector.load_local_dataset(file_path)
        
        if not eurepoc_raw.empty:
            # Analyze the raw dataset
            eurepoc_collector.analyze_dataset(eurepoc_raw)
            
            # Process the data
            eurepoc_processed = eurepoc_collector.process_incidents(eurepoc_raw)
            
            # Save processed data
            output_filename = "eurepoc_processed.csv"
            eurepoc_processed.to_csv(processed_data_dir / output_filename, index=False)
            
            print(f"\nEuRepoC Processing Complete:")
            print(f"Original records: {len(eurepoc_raw)}")
            print(f"Processed records: {len(eurepoc_processed)}")
            print(f"Output columns: {list(eurepoc_processed.columns)}")
            print(f"Saved to: {processed_data_dir / output_filename}")
            
            # Show sample of the processed data
            if len(eurepoc_processed) > 0:
                print("\nSample of processed data:")
                sample_cols = [col for col in ['title', 'date', 'incident_type', 'receiver_country', 'severity'] 
                             if col in eurepoc_processed.columns]
                print(eurepoc_processed[sample_cols].head(3))
        else:
            print(f"Failed to process {file_path.name}")

if __name__ == "__main__":
    main()