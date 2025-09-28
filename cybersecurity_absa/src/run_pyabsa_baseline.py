# src/run_pyabsa_baseline.py
import warnings
import os
import logging
from pyabsa import AspectTermExtraction as ATEPC
import configparser
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# === Suppress warnings and logs ===
# Set TensorFlow environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DEPRECATION_WARNINGS'] = '0'

# Configure warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*tf\\.losses\\.sparse_softmax_cross_entropy.*")
warnings.filterwarnings("ignore", message=".*Importing 'parser.split_arg_string'.*")

# Configure logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('pyabsa').setLevel(logging.ERROR)
logging.getLogger('spacy').setLevel(logging.ERROR)
logging.getLogger('weasel').setLevel(logging.ERROR)

def fix_tokenizer(tokenizer):
    """Monkey-patch the tokenizer to add missing attributes"""
    # Check if the tokenizer has the _bos_token attribute but not bos_token
    if hasattr(tokenizer, '_bos_token') and not hasattr(tokenizer, 'bos_token'):
        tokenizer.bos_token = tokenizer._bos_token
    
    if hasattr(tokenizer, '_eos_token') and not hasattr(tokenizer, 'eos_token'):
        tokenizer.eos_token = tokenizer._eos_token
    
    if hasattr(tokenizer, '_sep_token') and not hasattr(tokenizer, 'sep_token'):
        tokenizer.sep_token = tokenizer._sep_token
    
    if hasattr(tokenizer, '_cls_token') and not hasattr(tokenizer, 'cls_token'):
        tokenizer.cls_token = tokenizer._cls_token
    
    if hasattr(tokenizer, '_pad_token') and not hasattr(tokenizer, 'pad_token'):
        tokenizer.pad_token = tokenizer._pad_token
    
    if hasattr(tokenizer, '_unk_token') and not hasattr(tokenizer, 'unk_token'):
        tokenizer.unk_token = tokenizer._unk_token
    
    return tokenizer

def run_pyabsa_baseline():
    # --- Resolve project root and config ---
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.ini'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if 'paths' not in config:
        raise ValueError("Missing '[paths]' section in config.ini")
    
    # --- Resolve directories ---
    try:
        processed_data_dir = project_root / config['paths']['processed_data_dir']
    except KeyError as e:
        raise KeyError(f"Missing key in config.ini: {e}")
    
    # Ensure directories exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset with BERTopic assignments (corrected filename)
    input_file = processed_data_dir / 'combined_dataset_with_topics.csv'
    
    print(f"Looking for input file at: {input_file.absolute()}")
    
    if not input_file.exists():
        # Check if there are similar files in the directory
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

    # Monkey-patch the tokenizer before initializing PyABSA
    from transformers import DebertaV2TokenizerFast
    
    # Save the original __init__ method
    original_init = DebertaV2TokenizerFast.__init__
    
    # Define a new __init__ method that adds the missing attributes
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        fix_tokenizer(self)
    
    # Replace the __init__ method
    DebertaV2TokenizerFast.__init__ = patched_init

    # Initialize PyABSA (using the correct API from documentation)
    print("Loading PyABSA model...")
    try:
        # Use the correct initialization method from the documentation
        aspect_extractor = ATEPC.AspectExtractor('english')
        print("Successfully loaded PyABSA model with 'english' checkpoint")
    except Exception as e:
        print(f"Error loading 'english' checkpoint: {e}")
        try:
            # Try with 'multilingual' checkpoint as fallback
            aspect_extractor = ATEPC.AspectExtractor('multilingual')
            print("Successfully loaded PyABSA model with 'multilingual' checkpoint")
        except Exception as e2:
            print(f"Error loading 'multilingual' checkpoint: {e2}")
            raise RuntimeError("Failed to load any PyABSA checkpoint") from e2

    # Sample texts for baseline extraction (limit for initial run)
    sample_texts = df_with_topics['clean_text'].head(20).tolist()  # Adjust sample size as needed

    results = []
    print("Extracting aspects using PyABSA baseline...")
    for i, text in enumerate(tqdm(sample_texts, desc="Extracting aspects")):
        try:
            # Extract aspects
            prediction = aspect_extractor.predict(
                text,
                save_result=False,
                ignore_error=True
            )
            
            # Process results based on PyABSA output structure
            # The prediction result structure might vary, so we need to handle it carefully
            if hasattr(prediction, 'aspect'):
                aspects = prediction.aspect
            elif isinstance(prediction, dict) and 'aspect' in prediction:
                aspects = prediction['aspect']
            else:
                aspects = []
                
            if hasattr(prediction, 'sentiment'):
                sentiments = prediction.sentiment
            elif isinstance(prediction, dict) and 'sentiment' in prediction:
                sentiments = prediction['sentiment']
            else:
                sentiments = []
                
            if hasattr(prediction, 'confidence'):
                confidences = prediction.confidence
            elif isinstance(prediction, dict) and 'confidence' in prediction:
                confidences = prediction['confidence']
            else:
                confidences = []
            
            aspect_data = {
                'text_id': i,
                'original_text': text,
                'aspects': aspects,
                'sentiments': sentiments,
                'confidences': confidences,
                'success': True
            }
            results.append(aspect_data)
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

    # Save baseline results
    output_file = processed_data_dir / 'baseline_aspect_extraction.csv'
    baseline_results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    # Analyze results (basic)
    print(f"\nBaseline Analysis Results:")
    print(f"Success Rate: {baseline_results_df['success'].mean():.2%}")
    
    successful_df = baseline_results_df[baseline_results_df['success']]
    if len(successful_df) > 0:
        all_aspects = [aspect for sublist in successful_df['aspects'] for aspect in sublist]
        if all_aspects:
            unique_aspects = pd.Series(all_aspects).value_counts()
            print(f"Top 5 extracted aspects: {unique_aspects.head(5).to_dict()}")
        
        all_sentiments = [sentiment for sublist in successful_df['sentiments'] for sentiment in sublist]
        if all_sentiments:
            sentiment_dist = pd.Series(all_sentiments).value_counts()
            print(f"Sentiment distribution: {sentiment_dist.to_dict()}")
    
    # Show failed extractions
    failed_df = baseline_results_df[~baseline_results_df['success']]
    if len(failed_df) > 0:
        print(f"\nFailed extractions: {len(failed_df)}")
        for _, row in failed_df.head(3).iterrows():
            print(f"  - Error: {row['error'][:100]}...")

    return baseline_results_df

def main():
    try:
        baseline_results_df = run_pyabsa_baseline()
        print("\n PyABSA baseline analysis completed successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the BERTopic analysis script first.")
    except Exception as e:
        print(f"Error during PyABSA baseline analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()