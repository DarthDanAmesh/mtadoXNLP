# src/collect_csis_trafilatura.py
from trafilatura import fetch_url, extract, extract_metadata
import pandas as pd
import time
import configparser
from datetime import datetime
from pathlib import Path

class CSISReportsCollectorTrafilatura:
    def __init__(self):
        # Read configuration from project root
        self.config = configparser.ConfigParser()
        self.project_root = Path(__file__).resolve().parent.parent  # cybersecurity_absa/
        config_path = self.project_root / 'config.ini'
        self.config.read(config_path)
        
        self.base_url = "https://www.csis.org/programs/technology-policy/programs/cyber-policy"
        # For demonstration, let's use a single known report URL
        # In practice, you'd need to scrape the main page to get a list of report URLs
        self.report_urls = [
            "https://www.csis.org/analysis/china%E2%80%99s-cyber-operations-south-china-sea",
            # Add more URLs here or fetch them dynamically
        ]

    def collect_reports(self):
        """Collect CSIS reports using Trafilatura"""
        all_reports = []
        
        if not self.report_urls:
            print("No URLs provided for collection")
            return pd.DataFrame(all_reports)
            
        for i, url in enumerate(self.report_urls[:3]): # Limit for testing
            print(f"Extracting report {i+1}/{len(self.report_urls[:3])}: {url}")
            try:
                downloaded = fetch_url(url)
                if downloaded:
                    # Create trafilatura config from config.ini
                    trafilatura_config = {
                        'output_format': self.config['trafilatura']['output_format'],
                        'with_metadata': self.config['trafilatura'].getboolean('with_metadata'),
                        'include_tables': self.config['trafilatura'].getboolean('include_tables'),
                        'include_images': self.config['trafilatura'].getboolean('include_images')
                    }
                    
                    content = extract(downloaded, **trafilatura_config)
                    metadata = extract_metadata(downloaded)

                    if content and content.get('text'):
                        report_data = {
                            'source': 'CSIS_Trafilatura',
                            'url': url,
                            'title': content.get('title', metadata.get('title', 'No Title') if metadata else 'No Title'),
                            'content_text': content.get('text', ''),
                            'date_collected': datetime.now().isoformat(),
                            'extraction_success': True,
                            'metadata_full': metadata.to_dict() if metadata else {} # Store as dict
                        }
                        all_reports.append(report_data)
                        print(f"Successfully extracted: {report_data['title'][:50]}...")
                    else:
                        print(f"No content extracted from {url}")
                else:
                    print(f"Failed to download {url}")
            except Exception as e:
                print(f"Error processing {url}: {e}")
                # Add a failed record for tracking
                all_reports.append({
                    'source': 'CSIS_Trafilatura',
                    'url': url,
                    'title': 'Failed Extraction',
                    'content_text': '',
                    'date_collected': datetime.now().isoformat(),
                    'extraction_success': False,
                    'error': str(e)
                })
            time.sleep(2)  # Respect server rate limits
            
        return pd.DataFrame(all_reports)

def main():
    try:
        # Initialize collector
        csis_collector_trafilatura = CSISReportsCollectorTrafilatura()
        
        # Get paths from config relative to project root
        processed_data_dir = csis_collector_trafilatura.project_root / csis_collector_trafilatura.config['paths']['processed_data_dir']
        
        # Ensure directory exists
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute collection
        print("Starting CSIS report collection...")
        csis_trafilatura_reports = csis_collector_trafilatura.collect_reports()

        if not csis_trafilatura_reports.empty:
            # Calculate success rate
            success_rate = csis_trafilatura_reports['extraction_success'].mean() * 100
            successful_reports = csis_trafilatura_reports[csis_trafilatura_reports['extraction_success']]
            
            # Save data
            csis_trafilatura_reports.to_csv(processed_data_dir / 'csis_trafilatura_processed.csv', index=False)
            print(f"CSIS Trafilatura: {len(successful_reports)}/{len(csis_trafilatura_reports)} reports successfully collected ({success_rate:.1f}% success rate)")
            
            # Show sample of successful extractions
            if len(successful_reports) > 0:
                print("\nSample of collected reports:")
                for i, row in successful_reports.head(2).iterrows():
                    print(f"  - {row['title'][:60]}...")
        else:
            print("No reports were collected")
            
    except KeyError as e:
        print(f"Configuration error: {e}")
        print("Please check your config.ini file has the required sections")
    except Exception as e:
        print(f"Unexpected error in CSIS collection: {e}")

if __name__ == "__main__":
    main()