# src/preprocess_data.py
import pandas as pd
import re
import configparser
from pathlib import Path

class DataPreprocessor:
    def __init__(self):
        # Resolve project root: cybersecurity_absa/
        self.project_root = Path(__file__).parent.parent
        config_path = self.project_root / 'config.ini'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # Validate config has 'paths' section
        if 'paths' not in self.config:
            raise ValueError("Missing '[paths]' section in config.ini")
        
        # Resolve data directories
        try:
            self.raw_data_dir = self.project_root / self.config['paths']['raw_data_dir']
            self.processed_data_dir = self.project_root / self.config['paths']['processed_data_dir']
        except KeyError as e:
            raise KeyError(f"Missing key in config.ini [paths]: {e}")

        self.cybersecurity_terms = [
            'firewall', 'intrusion detection', 'patch', 'vulnerability', 'breach',
            'ransomware', 'phishing', 'malware', 'encryption', 'authentication',
            'incident response', 'security controls', 'threat intelligence'
        ]

    def clean_text(self, text):
        """Clean and normalize text data"""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        # Keep letters, digits, whitespace, hyphens, and periods (for terms like "zero-day")
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove basic stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        text = ' '.join(word for word in text.split() if word not in stop_words)
        return text

    def preprocess_dataframe(self, df, source_name="unknown"):
        """Preprocess a cybersecurity dataframe"""
        if df.empty:
            print(f"Warning: Empty dataframe for source '{source_name}'")
            return df

        df_processed = df.copy()
        
        # Determine text column
        text_col = None
        for candidate in ['content_text', 'description', 'text', 'summary']:
            if candidate in df_processed.columns:
                text_col = candidate
                break
        
        if text_col is None:
            print(f"Warning: No suitable text column found in {source_name}. Available columns: {list(df.columns)}")
            df_processed['clean_text'] = ""
            df_processed['cyber_terms'] = [[] for _ in range(len(df_processed))]
            df_processed['text_length'] = 0
            df_processed['cyber_term_count'] = 0
            return df_processed

        # Clean text
        df_processed['clean_text'] = df_processed[text_col].apply(self.clean_text)
        
        # Extract cybersecurity terms
        def extract_cyber_terms(text):
            if not isinstance(text, str):
                return []
            return [term for term in self.cybersecurity_terms if term in text]

        df_processed['cyber_terms'] = df_processed['clean_text'].apply(extract_cyber_terms)
        df_processed['text_length'] = df_processed['clean_text'].str.len()
        df_processed['cyber_term_count'] = df_processed['cyber_terms'].apply(len)

        # Filter out very short texts (< 50 characters)
        initial_count = len(df_processed)
        df_processed = df_processed[df_processed['text_length'] > 50].copy()
        filtered_count = len(df_processed)
        if filtered_count < initial_count:
            print(f"Filtered out {initial_count - filtered_count} short records from {source_name}")

        return df_processed

    def merge_datasets(self, dfs):
        """Merge multiple dataframes, deduplicate by clean_text"""
        if not dfs:
            return pd.DataFrame()
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if 'clean_text' in combined_df.columns:
            before = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['clean_text'])
            after = len(combined_df)
            if before != after:
                print(f"Removed {before - after} duplicate records based on clean_text")
        
        combined_df = combined_df.reset_index(drop=True)
        return combined_df

    def create_sample_data(self):
        """Create sample cybersecurity data for testing (fallback only)"""
        print("Creating SAMPLE data for testing (no real data found)...")
        
        sample_data = [
            {
                'id': 0,
                'title': 'Ransomware Attack on Healthcare System',
                'content_text': 'A major ransomware attack targeted a healthcare system, encrypting patient records and demanding payment. The attack exploited vulnerabilities in the firewall configuration.',
                'source': 'sample',
                'date': '2024-01-15'
            },
            {
                'title': 'Phishing Campaign Targets Financial Institutions',
                'content_text': 'A sophisticated phishing campaign targeted multiple financial institutions, using social engineering to bypass authentication systems. Incident response teams were activated.',
                'source': 'sample',
                'date': '2024-01-20'
            },
            {
                'title': 'Vulnerability in Encryption Software Discovered',
                'content_text': 'Security researchers discovered a critical vulnerability in widely used encryption software that could allow threat actors to bypass security controls.',
                'source': 'sample',
                'date': '2024-01-25'
            },
            {
                'title': 'Malware Infection via Supply Chain Attack',
                'content_text': 'A supply chain attack resulted in malware being distributed through legitimate software updates. Intrusion detection systems failed to identify the threat initially.',
                'source': 'sample',
                'date': '2024-02-01'
            },
            {
                'title': 'Data Breach Exposes Customer Information',
                'content_text': 'A data breach at a major corporation exposed sensitive customer information. The breach was caused by inadequate patch management and weak security controls.',
                'source': 'sample',
                'date': '2024-02-05'
            }
        ]
        return pd.DataFrame(sample_data)


def main():
    try:
        preprocessor = DataPreprocessor()
        
        # Ensure output directory exists
        preprocessor.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define expected input files (from earlier collection steps)
        dataset_files = {
            'EuRepoC': 'eurepoc_processed.csv',
            'CISA': 'cisa_trafilatura_processed.csv',
            'CSIS': 'csis_trafilatura_processed.csv'
        }

        print("Loading processed datasets...")
        datasets = []

        for source_name, filename in dataset_files.items():
            file_path = preprocessor.processed_data_dir / filename
            if file_path.exists() and file_path.stat().st_size > 0:
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        print(f"{source_name}: {len(df)} records loaded")
                        datasets.append((source_name, df))
                    else:
                        print(f"{source_name}: File is empty")
                except Exception as e:
                    print(f"{source_name}: Error reading {filename} - {e}")
            else:
                print(f"{source_name}: File not found or empty ({file_path})")

        # Fallback to sample data if nothing loaded
        if not datasets:
            print("\n❗ No real datasets found. Falling back to sample data for testing.")
            sample_df = preprocessor.create_sample_data()
            datasets = [('Sample', sample_df)]

        # Preprocess each dataset
        print("\nPreprocessing datasets...")
        processed_dfs = []
        for source_name, df in datasets:
            processed_df = preprocessor.preprocess_dataframe(df, source_name=source_name)
            if not processed_df.empty:
                processed_dfs.append(processed_df)
            else:
                print(f"{source_name}: No valid records after preprocessing")

        # Merge and save
        if processed_dfs:
            combined_df = preprocessor.merge_datasets(processed_dfs)
            output_path = preprocessor.processed_data_dir / 'combined_dataset_phase1.csv'
            combined_df.to_csv(output_path, index=False)
            print(f"\nCombined dataset saved to: {output_path}")
            print(f"Final dataset size: {len(combined_df)} records")
            
            # Print summary stats
            if 'source' in combined_df.columns:
                print(f"Sources: {dict(combined_df['source'].value_counts())}")
            print(f"Avg. text length: {combined_df['text_length'].mean():.1f} chars")
            print(f"Avg. cyber terms/doc: {combined_df['cyber_term_count'].mean():.1f}")
        else:
            print("No data to save after preprocessing.")

    except Exception as e:
        print(f"Fatal error in preprocessing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()