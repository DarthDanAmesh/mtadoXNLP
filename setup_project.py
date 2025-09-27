import os
from pathlib import Path

# Create project directory structure
project_dir = Path("cybersecurity_absa")
project_dir.mkdir(exist_ok=True)

subdirs = ['data/raw', 'data/processed', 'models', 'notebooks', 'src', 'reports', 'models/visualizations']
for subdir in subdirs:
    (project_dir / subdir).mkdir(exist_ok=True, parents=True)

# Create configuration file
config_content = """  
[paths]  
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

with open(project_dir / 'config.ini', 'w') as f:
    f.write(config_content)
print("Project structure and configuration initialized")