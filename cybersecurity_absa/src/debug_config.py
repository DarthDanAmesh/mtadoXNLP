# src/debug_config.py
import configparser
from pathlib import Path
import os

def debug_config():
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if config.ini exists in project root
    config_path = Path(__file__).parent.parent / 'config.ini'
    print(f"Expected config path: {config_path}")
    print(f"Config file exists: {config_path.exists()}")
    
    if config_path.exists():
        # Read and display config contents
        config = configparser.ConfigParser()
        config.read(config_path)
        
        print(f"Config sections: {config.sections()}")
        for section in config.sections():
            print(f"\n[{section}]")
            for key, value in config.items(section):
                print(f"  {key} = {value}")
    else:
        print("Config file not found! Creating a basic one...")
        # Create a basic config file
        config_content = """[paths]
data_dir = data
raw_data_dir = data/raw
processed_data_dir = data/processed
models_dir = models
reports_dir = reports

[models]
pretrained_bert = microsoft/deberta-v3-base
max_seq_len = 128
dropout = 0.1
batch_size = 16

[topic_modeling]
embedding_model = all-mpnet-base-v2
min_topic_size = 10
nr_topics = 5
verbose = True

[web_scraping]
target_language = en
include_comments = False
include_tables = False
favor_precision = True

[trafilatura]
output_format = json
with_metadata = True
include_tables = False
include_images = False
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
        print("Basic config file created!")

if __name__ == "__main__":
    debug_config()