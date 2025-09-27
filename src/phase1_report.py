# src/phase1_report.py
import configparser
import pandas as pd
from datetime import datetime
from pathlib import Path

def generate_phase1_report():
    # Read configuration with correct path
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent / 'config.ini'
    config.read(config_path)
    
    # Get paths from config
    processed_data_dir = Path(config['paths']['processed_data_dir'])
    reports_dir = Path(config['paths']['reports_dir'])
    
    # Ensure reports directory exists
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load key outputs
        combined_df = pd.read_csv(processed_data_dir / 'dataset_with_bertopics.csv')
        baseline_results_df = pd.read_csv(processed_data_dir / 'baseline_aspect_extraction.csv')
        
        # Calculate metrics
        total_records = len(combined_df)
        unique_topics = combined_df['bertopic_id'].nunique()
        success_rate = baseline_results_df['success'].mean()
        
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please run the previous analysis scripts first.")
        return

    report = f"""
    PHASE 1: FOUNDATION BUILDING - COMPLETION REPORT
    =============================================
    EXECUTION PERIOD: Weeks 1-4
    COMPLETION DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    KEY ACHIEVEMENTS:
    - Environment configured with PyABSA, BERTopic, and Trafilatura
    - Collected data from EuRepoC, CISA (Trafilatura), CSIS (Trafilatura)
    - Generated {total_records} processed records
    - Performed BERTopic semantic clustering ({unique_topics} unique topics identified)
    - Conducted baseline PyABSA aspect extraction on {len(baseline_results_df)} samples
    - Generated interactive BERTopic visualizations
    TECHNICAL METRICS:
    - Total Records Processed: {total_records}
    - BERTopic Topics Identified: {unique_topics}
    - PyABSA Baseline Success Rate: {success_rate:.2%}
    CHALLENGES ADDRESSED:
    - Integrated Trafilatura for high-quality content extraction
    - Implemented BERTopic for superior semantic topic modeling
    - Established baseline PyABSA pipeline
    - Created structured data pipeline
    NEXT PHASE PREPARATION:
    1. Fine-tune PyABSA model on the full corpus
    2. Develop explainability layer (LIME/SHAP)
    3. Construct knowledge graph based on topics, aspects, and entities
    4. Begin Flask tool development incorporating BERTopic and PyABSA outputs
    CONCLUSION: Phase 1 successfully established a robust foundation
    for advanced cybersecurity aspect analysis using state-of-the-art tools
    (Trafilatura, BERTopic, PyABSA).
    """
    
    report_path = reports_dir / 'phase1_completion_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(report)
    print(f"Report saved to {report_path}")

def main():
    try:
        generate_phase1_report()
    except Exception as e:
        print(f"Error generating Phase 1 report: {e}")

if __name__ == "__main__":
    main()