# === PATCH DebertaV2TokenizerFast to avoid bos_token/eos_token error ===
from transformers import DebertaV2TokenizerFast

_original_getattribute = DebertaV2TokenizerFast.__getattribute__

def _safe_getattribute(self, name):
    if name == 'bos_token':
        return getattr(self, '_bos_token', None) or '[CLS]'
    if name == 'eos_token':
        return getattr(self, '_eos_token', None) or '[SEP]'
    return _original_getattribute(self, name)

DebertaV2TokenizerFast.__getattribute__ = _safe_getattribute

# === Imports ===
import warnings
import os
import logging
from pyabsa import AspectTermExtraction as ATEPC
from pyabsa import available_checkpoints
import configparser
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# === Suppress warnings early ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('pyabsa').setLevel(logging.ERROR)
logging.getLogger('spacy').setLevel(logging.ERROR)

# === Load a working checkpoint ONCE ===
atepc_ckpts = available_checkpoints().get("ATEPC", [])
print("Available ATEPC checkpoints:", atepc_ckpts)

# These don't exist for ATEPC
safe_candidates = ['bert_base', 'lcf_bert', 'fast_lcf_bert', 'roberta_base']
candidates = [c for c in safe_candidates if c in atepc_ckpts] + ['english', 'multilingual']

aspect_extractor = None
for ckpt in candidates:
    try:
        print(f"Loading checkpoint: {ckpt}")
        aspect_extractor = ATEPC.AspectExtractor(ckpt)
        print(f"Success with {ckpt}")
        break
    except Exception as e:
        print(f"Failed to load checkpoint '{ckpt}': {e}")

if aspect_extractor is None:
    raise RuntimeError("No working ATEPC checkpoint found. Please check available_checkpoints().")

def run_pyabsa_baseline(aspect_extractor):
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.ini'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if 'paths' not in config:
        raise ValueError("Missing '[paths]' section in config.ini")
    
    try:
        processed_data_dir = project_root / config['paths']['processed_data_dir']
    except KeyError as e:
        raise KeyError(f"Missing key in config.ini: {e}")
    
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = processed_data_dir / 'combined_dataset_with_topics.csv'
    print(f"Looking for input file at: {input_file.absolute()}")
    
    if not input_file.exists():
        existing_files = list(processed_data_dir.glob("*.csv"))
        print(f"Available CSV files in {processed_data_dir}:")
        for file in existing_files:
            print(f"  - {file.name}")
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Please run the BERTopic analysis script first."
        )
    
    df_with_topics = pd.read_csv(input_file)
    print(f"Successfully loaded data with {len(df_with_topics)} records")

    sample_texts = df_with_topics['clean_text'].head(20).tolist()

    results = []
    print("Extracting aspects using PyABSA baseline...")
    for i, text in enumerate(tqdm(sample_texts, desc="Extracting aspects")):
        try:
            prediction = aspect_extractor.predict(
                text,
                save_result=False,
                ignore_error=True
            )
            
            aspects = getattr(prediction, 'aspect', [])
            sentiments = getattr(prediction, 'sentiment', [])
            confidences = getattr(prediction, 'confidence', [])
            
            results.append({
                'text_id': i,
                'original_text': text,
                'aspects': aspects,
                'sentiments': sentiments,
                'confidences': confidences,
                'success': True
            })
        except Exception as e:
            results.append({
                'text_id': i,
                'original_text': text,
                'aspects': [],
                'sentiments': [],
                'confidences': [],
                'success': False,
                'error': str(e)
            })

    baseline_results_df = pd.DataFrame(results)
    print(f"Baseline PyABSA extraction completed for {len(baseline_results_df)} samples.")

    output_file = processed_data_dir / 'baseline_aspect_extraction.csv'
    baseline_results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    print(f"\nBaseline Analysis Results:")
    print(f"Success Rate: {baseline_results_df['success'].mean():.2%}")
    
    successful_df = baseline_results_df[baseline_results_df['success']]
    if len(successful_df) > 0:
        all_aspects = [a for sublist in successful_df['aspects'] for a in sublist]
        if all_aspects:
            top_aspects = pd.Series(all_aspects).value_counts().head(5).to_dict()
            print(f"Top 5 extracted aspects: {top_aspects}")
        
        all_sentiments = [s for sublist in successful_df['sentiments'] for s in sublist]
        if all_sentiments:
            sentiment_dist = pd.Series(all_sentiments).value_counts().to_dict()
            print(f"Sentiment distribution: {sentiment_dist}")

    return baseline_results_df


def main():
    try:
        baseline_results_df = run_pyabsa_baseline(aspect_extractor)
        print("\nPyABSA baseline analysis completed successfully.")
    except Exception as e:
        print(f"Error during PyABSA baseline analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
