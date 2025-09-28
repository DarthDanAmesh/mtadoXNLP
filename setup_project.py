from pathlib import Path

# Get current directory (repo root, assumed to have src/ and config.ini already)
root_dir = Path.cwd()

# Main project directory to create folders inside (cybersecurity_absa folder)
project_dir = root_dir / "cybersecurity_absa"
project_dir.mkdir(exist_ok=True)

# Subdirectories to create under cybersecurity_absa/
subdirs = [
    'data/raw',
    'data/processed',
    'models',
    'models/visualizations',
    'notebooks',
    'reports',
    # 'src' is intentionally excluded
]

# Create subdirectories if they don’t already exist
for subdir in subdirs:
    subdir_path = project_dir / subdir
    if not subdir_path.exists():
        subdir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {subdir_path}")
    else:
        print(f"Exists: {subdir_path}")

# Don't recreate config.ini if it already exists in the root directory
config_path = root_dir / 'config.ini'  # This is the correct location for config.ini
if config_path.exists():
    print(f"Skipped: config.ini already exists at {config_path}")
else:
    config_content = """\
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
    with open(config_path, 'w') as f:
        f.write(config_content.strip() + '\n')
    print(f"Created: config.ini at {config_path}")

# Optional: Confirm that src/ is found at the root level
src_path = root_dir / 'src'
if src_path.exists() and src_path.is_dir():
    print(f"Found existing src/ directory at: {src_path}")
else:
    print("'src/' directory not found at root. Check your clone or repo structure.")
