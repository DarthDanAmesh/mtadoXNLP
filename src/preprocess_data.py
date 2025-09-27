# src/preprocess_data.py
import pandas as pd
import re
import configparser
from pathlib import Path

class DataPreprocessor:
    def __init__(self):
        # Read configuration with correct path
        self.config = configparser.ConfigParser()
        config_path = Path(__file__).parent.parent / 'config.ini'
        self.config.read(config_path)
        
        self.cybersecurity_terms = [
            'firewall', 'intrusion detection', 'patch', 'vulnerability', 'breach',
            'ransomware', 'phishing', 'malware', 'encryption', 'authentication',
            'incident response', 'security controls', 'threat intelligence'
        ]

    def clean_text(self, text):
        """Clean and normalize text data"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep cybersecurity-relevant symbols
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common non-informative words (basic stop words)
        stop_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text.strip()

    def preprocess_dataframe(self, df):
        """Preprocess cybersecurity dataframe"""
        if df.empty:
            return df
            
        df_processed = df.copy()
        # Clean text columns
        text_col = 'content_text' if 'content_text' in df_processed.columns else 'description'
        if text_col in df_processed.columns:
            df_processed['clean_text'] = df_processed[text_col].apply(self.clean_text)
        else:
            print(f"Text column '{text_col}' not found in dataframe.")
            return df_processed

        # Extract cybersecurity terminology
        def extract_cyber_terms(text):
            if not isinstance(text, str):
                return []
            return [term for term in self.cybersecurity_terms if term in text.lower()]

        df_processed['cyber_terms'] = df_processed['clean_text'].apply(extract_cyber_terms)

        # Add metadata
        df_processed['text_length'] = df_processed['clean_text'].str.len()
        df_processed['cyber_term_count'] = df_processed['cyber_terms'].apply(len)

        # Remove empty records
        df_processed = df_processed[df_processed['clean_text'].str.len() > 50]
        return df_processed

    def merge_datasets(self, dfs):
        """Merge multiple dataframes from different sources"""
        if not dfs:
            return pd.DataFrame()
        # Combine datasets
        combined_df = pd.concat(dfs, ignore_index=True)
        # Remove duplicates based on text content
        text_col = 'clean_text' if 'clean_text' in combined_df.columns else 'content_text'
        if text_col in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=[text_col])
        # Reset index
        combined_df = combined_df.reset_index(drop=True)
        return combined_df

    def create_sample_data(self):
        """Create sample cybersecurity data for testing"""
        print("Creating sample cybersecurity data for testing...")
        
        sample_data = []
        cybersecurity_incidents = [
            {
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
        
        for i, incident in enumerate(cybersecurity_incidents):
            sample_data.append({
                'id': i,
                'title': incident['title'],
                'description': incident['content_text'],
                'content_text': incident['content_text'],
                'source': incident['source'],
                'date': incident['date']
            })
        
        return pd.DataFrame(sample_data)

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Get paths from config
    processed_data_dir = Path(preprocessor.config['paths']['processed_data_dir'])
    
    # Ensure directory exists
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = []
    dataset_names = ['EuRepoC', 'CISA', 'CSIS']
    dataset_files = ['eurepoc_processed.csv', 'cisa_trafilatura_processed.csv', 'csis_trafilatura_processed.csv']
    
    print("Loading datasets...")
    
    for name, filename in zip(dataset_names, dataset_files):
        file_path = processed_data_dir / filename
        try:
            # Check if file exists and has content
            if file_path.exists() and file_path.stat().st_size > 0:
                df = pd.read_csv(file_path)
                if not df.empty:
                    print(f"{name}: {len(df)} records")
                    datasets.append((name, df))
                else:
                    print(f"{name}: File exists but is empty")
            else:
                print(f"{name}: File not found or empty")
        except Exception as e:
            print(f"{name}: Error loading - {e}")
    
    # If no datasets were successfully loaded, create sample data
    if not datasets:
        print("No valid datasets found. Creating sample data for testing...")
        sample_df = preprocessor.create_sample_data()
        datasets = [('Sample', sample_df)]
    
    # Preprocess datasets
    processed_datasets = []
    for name, df in datasets:
        print(f"Preprocessing {name} data...")
        processed_df = preprocessor.preprocess_dataframe(df)
        if not processed_df.empty:
            processed_datasets.append(processed_df)
            print(f"{name}: {len(processed_df)} records after preprocessing")
        else:
            print(f"{name}: No valid records after preprocessing")
    
    # Merge datasets
    if processed_datasets:
        combined_df = preprocessor.merge_datasets(processed_datasets)
        # Save processed data
        combined_df.to_csv(processed_data_dir / 'combined_dataset_phase1.csv', index=False)
        print(f"Data preprocessing and merging completed: {len(combined_df)} records")
        
        # Show basic statistics
        print(f"Sources in combined data: {combined_df['source'].value_counts().to_dict()}")
        print(f"Average text length: {combined_df['text_length'].mean():.1f} characters")
        print(f"Average cyber terms per document: {combined_df['cyber_term_count'].mean():.1f}")
    else:
        print("No valid data to process after preprocessing")

if __name__ == "__main__":
    main()