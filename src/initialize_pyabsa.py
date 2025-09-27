# src/initialize_pyabsa.py
from pyabsa import AspectTermExtraction as ATEPC
import configparser
import os
from pathlib import Path

# Add BERTopic imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired

# Load configuration with correct path
config = configparser.ConfigParser()
config_path = Path(__file__).parent.parent / 'config.ini'  # Go up one level from src/
config.read(config_path)

def initialize_pyabsa():
    # Initialize PyABSA with cybersecurity-optimized settings
    pyabsa_config = ATEPC.ATEPCConfigManager.get_atepc_config_english()

    # Critical cybersecurity-specific adjustments
    pyabsa_config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC
    pyabsa_config.pretrained_bert = config['models']['pretrained_bert']
    pyabsa_config.max_seq_len = int(config['models']['max_seq_len'])
    pyabsa_config.dropout = float(config['models']['dropout'])
    pyabsa_config.eval_batch_size = int(config['models']['batch_size'])
    print("PyABSA Configuration prepared")

    # Initialize aspect extractor
    aspect_extractor = ATEPC.AspectExtractor(checkpoint="english")
    return aspect_extractor, pyabsa_config

# Test the configuration
try:
    aspect_extractor, pyabsa_config = initialize_pyabsa()
    print(f"PyABSA environment configured successfully")
    print(f"Using model: {config['models']['pretrained_bert']}")
except KeyError as e:
    print(f"Configuration error: {e}")
    print(f"Available sections: {config.sections()}")
    print(f"Config file path: {config_path}")
    print(f"Current working directory: {os.getcwd()}")