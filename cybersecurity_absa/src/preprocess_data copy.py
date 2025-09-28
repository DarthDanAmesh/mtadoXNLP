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

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Get paths from config
    processed_data_dir = Path(preprocessor.config['paths']['processed_data_dir'])
    
    # Ensure directory exists
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load collected data
        eurepoc_df = pd.read_csv(processed_data_dir / 'eurepoc_processed.csv')
        cisa_df = pd.read_csv(processed_data_dir / 'cisa_trafilatura_processed.csv')
        csis_df = pd.read_csv(processed_data_dir / 'csis_trafilatura_processed.csv')
        
        print("Loaded datasets:")
        print(f"EuRepoC: {len(eurepoc_df)} records")
        print(f"CISA: {len(cisa_df)} records")
        print(f"CSIS: {len(csis_df)} records")
        
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please run the data collection scripts first.")
        return

    # Preprocess individual datasets
    eurepoc_processed = preprocessor.preprocess_dataframe(eurepoc_df)
    cisa_processed = preprocessor.preprocess_dataframe(cisa_df)
    csis_processed = preprocessor.preprocess_dataframe(csis_df)

    # Merge datasets
    combined_df = preprocessor.merge_datasets([eurepoc_processed, cisa_processed, csis_processed])

    # Save processed data
    combined_df.to_csv(processed_data_dir / 'combined_dataset_phase1.csv', index=False)
    print(f"Data preprocessing and merging completed: {len(combined_df)} records")

if __name__ == "__main__":
    main()