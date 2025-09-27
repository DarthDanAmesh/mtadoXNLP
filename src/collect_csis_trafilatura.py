# src/collect_csis_trafilatura.py
from trafilatura import fetch_url, bare_extraction
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
        
        self.base_url = "https://www.csis.org/topics/cybersecurity"
        # Use current CSIS cybersecurity reports and articles
        self.report_urls = [
            "https://www.csis.org/analysis/why-congress-must-protect-cyber-sharing",
            "https://www.csis.org/analysis/channeling-augustus-agentic-offensive-information-operations",
            "https://www.csis.org/analysis/ensuring-cybersecurity-digital-public-infrastructure",
            "https://www.csis.org/analysis/cybersecurity-implications-ai-adoption",
            "https://www.csis.org/analysis/strategic-competition-cyberspace",
            "https://www.csis.org/analysis/ransomware-resilience-critical-infrastructure",
            "https://www.csis.org/analysis/cyber-deterrence-21st-century",
            "https://www.csis.org/analysis/zero-trust-architecture-implementation",
            "https://www.csis.org/analysis/cyber-threat-intelligence-sharing",
            "https://www.csis.org/analysis/quantum-computing-cybersecurity-implications"
        ]

    def collect_reports(self):
        """Collect CSIS reports using Trafilatura's bare_extraction"""
        all_reports = []
        
        if not self.report_urls:
            print("No URLs provided for collection")
            return pd.DataFrame(all_reports)
            
        for i, url in enumerate(self.report_urls[:8]): # Limit for testing
            print(f"Extracting report {i+1}/{len(self.report_urls[:8])}: {url}")
            try:
                downloaded = fetch_url(url)
                if downloaded:
                    # Use bare_extraction which returns a dict with metadata
                    content_doc = bare_extraction(downloaded, url=url, with_metadata=True)
                    content = content_doc.as_dict() if content_doc else None
                    
                    if content and content.get('text'):
                        report_data = {
                            'source': 'CSIS_Trafilatura',
                            'url': url,
                            'title': content.get('title', 'No Title'),
                            'content_text': content.get('text', ''),
                            'author': content.get('author', ''),
                            'date': content.get('date', ''),
                            'description': content.get('description', ''),
                            'sitename': content.get('sitename', ''),
                            'categories': content.get('categories', ''),
                            'tags': content.get('tags', ''),
                            'date_collected': datetime.now().isoformat(),
                            'extraction_success': True,
                            'metadata_full': content  # Store the full extraction dict
                        }
                        all_reports.append(report_data)
                        print(f"Successfully extracted: {report_data['title'][:50]}...")
                    else:
                        print(f"No content extracted from {url}")
                        # Record failed extraction
                        all_reports.append({
                            'source': 'CSIS_Trafilatura',
                            'url': url,
                            'title': 'Failed Extraction - No Content',
                            'content_text': '',
                            'date_collected': datetime.now().isoformat(),
                            'extraction_success': False,
                            'error': 'No content extracted'
                        })
                else:
                    print(f"Failed to download {url}")
                    # Record failed download
                    all_reports.append({
                        'source': 'CSIS_Trafilatura',
                        'url': url,
                        'title': 'Failed Extraction - Download Error',
                        'content_text': '',
                        'date_collected': datetime.now().isoformat(),
                        'extraction_success': False,
                        'error': 'Failed to download URL'
                    })
            except Exception as e:
                print(f"Error processing {url}: {e}")
                # Add a failed record for tracking
                all_reports.append({
                    'source': 'CSIS_Trafilatura',
                    'url': url,
                    'title': 'Failed Extraction - Exception',
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
            
            # Show detailed success information
            if len(successful_reports) > 0:
                print("\nSample of collected reports:")
                for i, row in successful_reports.head(3).iterrows():
                    print(f"  - Title: {row['title'][:60]}...")
                    print(f"    Author: {row.get('author', 'N/A')}")
                    print(f"    Date: {row.get('date', 'N/A')}")
                    print(f"    Text length: {len(row['content_text'])} characters")
                    print()
                    
            # Show failed extractions for debugging
            failed_reports = csis_trafilatura_reports[~csis_trafilatura_reports['extraction_success']]
            if len(failed_reports) > 0:
                print(f"\nFailed extractions: {len(failed_reports)}")
                for i, row in failed_reports.iterrows():
                    print(f"  - {row['url']}: {row.get('error', 'Unknown error')}")
        else:
            print("No reports were collected")
            
    except KeyError as e:
        print(f"Configuration error: {e}")
        print("Please check your config.ini file has the required sections")
    except Exception as e:
        print(f"Unexpected error in CSIS collection: {e}")

if __name__ == "__main__":
    main()