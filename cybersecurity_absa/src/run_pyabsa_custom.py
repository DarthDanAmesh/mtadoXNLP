# src/run_custom_pyabsa.py
import warnings
import os
import logging
from pyabsa import AspectTermExtraction as ATEPC
import configparser
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import re
import ast

# === Suppress warnings and logs ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DEPRECATION_WARNINGS'] = '0'

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*tf\\.losses\\.sparse_softmax_cross_entropy.*")
warnings.filterwarnings("ignore", message=".*Importing 'parser.split_arg_string'.*")

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('pyabsa').setLevel(logging.ERROR)
logging.getLogger('spacy').setLevel(logging.ERROR)
logging.getLogger('weasel').setLevel(logging.ERROR)

def find_custom_model():
    """Find the custom-trained model checkpoint in the project root"""
    # The checkpoints are saved in the project root, not in the models directory
    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root.parent / "checkpoints"  # Go up one more level
    
    print(f"Searching for models in: {checkpoints_dir.absolute()}")
    
    if not checkpoints_dir.exists():
        print(f" Checkpoints directory not found: {checkpoints_dir}")
        return None
    
    print(f" Found checkpoints directory: {checkpoints_dir}")
    
    # Find all checkpoint directories that match our custom model pattern
    custom_checkpoints = []
    for checkpoint_dir in checkpoints_dir.iterdir():
        if checkpoint_dir.is_dir():
            dir_name_lower = checkpoint_dir.name.lower()
            is_fast_lcf = "fast_lcf_atepc" in dir_name_lower
            is_custom = "custom" in dir_name_lower or "cybersecurity" in dir_name_lower or "dataset" in dir_name_lower
            
            if is_fast_lcf and is_custom:
                # Check if it contains the necessary files for ATEPC
                required_files = [
                    checkpoint_dir / "fast_lcf_atepc.config",
                    checkpoint_dir / "fast_lcf_atepc.state_dict", 
                    checkpoint_dir / "fast_lcf_atepc.tokenizer"
                ]
                
                # Check each required file
                missing_files = [req_file for req_file in required_files if not req_file.exists()]
                
                if not missing_files:
                    custom_checkpoints.append(checkpoint_dir)
    
    if not custom_checkpoints:
        print(" No valid custom model checkpoints found")
        return None
    
    # Sort by APC accuracy (extracted from directory name) to get the best model
    def extract_acc_from_name(dirname):
        match = re.search(r'apcacc_(\d+\.\d+)', str(dirname))
        if match:
            return float(match.group(1))
        return 0.0
    
    custom_checkpoints.sort(key=lambda x: extract_acc_from_name(x.name), reverse=True)
    
    # Return the best model
    best_model = custom_checkpoints[0]
    best_acc = extract_acc_from_name(best_model.name)
    print(f" Selected best model: {best_model.name} (APC Accuracy: {best_acc})")
    return best_model

def parse_pyabsa_result(prediction):
    """Parse PyABSA prediction result and extract aspects, sentiments, confidences, and positions"""
    # Initialize default values
    aspects = []
    sentiments = []
    confidences = []
    positions = []
    
    try:
        # PyABSA can return results in different formats
        # Try to get aspects
        if hasattr(prediction, 'aspect'):
            aspects = prediction.aspect
        elif isinstance(prediction, dict) and 'aspect' in prediction:
            aspects = prediction['aspect']
        
        # Try to get sentiments
        if hasattr(prediction, 'sentiment'):
            sentiments = prediction.sentiment
        elif isinstance(prediction, dict) and 'sentiment' in prediction:
            sentiments = prediction['sentiment']
        
        # Try to get confidences
        if hasattr(prediction, 'confidence'):
            confidences = prediction.confidence
        elif isinstance(prediction, dict) and 'confidence' in prediction:
            confidences = prediction['confidence']
        
        # Try to get positions
        if hasattr(prediction, 'position'):
            positions = prediction.position
        elif isinstance(prediction, dict) and 'position' in prediction:
            positions = prediction['position']
        
        # If we still don't have aspects, try to parse from the string representation
        if not aspects and hasattr(prediction, '__str__'):
            pred_str = str(prediction)
            # Look for patterns like "aspect:-1 Confidence:0.964"
            aspect_matches = re.findall(r'([^:<\s]+):(-?\d+)\s+Confidence:(\d+\.\d+)', pred_str)
            if aspect_matches:
                aspects = [match[0] for match in aspect_matches]
                sentiments = [match[1] for match in aspect_matches]
                confidences = [float(match[2]) for match in aspect_matches]
        
        # Ensure all lists have the same length
        min_length = min(len(aspects), len(sentiments), len(confidences))
        aspects = aspects[:min_length]
        sentiments = sentiments[:min_length]
        confidences = confidences[:min_length]
        
        # If positions is empty or different length, fill with None
        if len(positions) != min_length:
            positions = [None] * min_length
            
        return aspects, sentiments, confidences, positions
        
    except Exception as e:
        print(f"Error parsing prediction result: {e}")
        return [], [], [], []

def run_custom_pyabsa():
    # Load configuration
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent / 'config.ini'
    config.read(config_path)
    
    # Get paths
    project_root = Path(__file__).parent.parent
    processed_data_dir = project_root / config['paths']['processed_data_dir']
    
    print(f" Project root: {project_root}")
    
    # Load dataset with BERTopic assignments
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
    print(f" Successfully loaded data with {len(df_with_topics)} records")

    # Find the custom model checkpoint
    print("\n" + "="*60)
    print(" SEARCHING FOR CUSTOM MODEL")
    print("="*60)
    custom_model_path = find_custom_model()

    print("\n" + "="*60)
    print(" ATTEMPTING TO LOAD MODEL")
    print("="*60)
    
    if custom_model_path:
        print(f" Loading custom cybersecurity model from: {custom_model_path}")
        try:
            # Convert Path to string for PyABSA
            model_path_str = str(custom_model_path)
            print(f" Attempting to load model from: {model_path_str}")
            
            aspect_extractor = ATEPC.AspectExtractor(checkpoint=model_path_str)
            print(" Successfully loaded custom cybersecurity model")
        except Exception as e:
            print(f" Error loading custom model: {e}")
            print(" Falling back to pretrained model...")
            try:
                # Use the pretrained English model
                aspect_extractor = ATEPC.AspectExtractor('english')
                print(" Successfully loaded pretrained English model")
            except Exception as e2:
                print(f" Error loading pretrained model: {e2}")
                raise
    else:
        print(" No custom model found, using pretrained model...")
        try:
            aspect_extractor = ATEPC.AspectExtractor('english')
            print(" Successfully loaded pretrained English model")
        except Exception as e:
            print(f" Error loading pretrained model: {e}")
            raise

    # Sample texts for extraction
    sample_texts = df_with_topics['clean_text'].head(20).tolist()

    results = []
    print("\n" + "="*60)
    print(" EXTRACTING ASPECTS")
    print("="*60)
    print(f"Extracting aspects from {len(sample_texts)} sample texts...")
    
    for i, text in enumerate(tqdm(sample_texts, desc="Extracting aspects")):
        try:
            # Extract aspects
            prediction = aspect_extractor.predict(
                text,
                save_result=False,
                ignore_error=True
            )
            
            # Parse the prediction result
            aspects, sentiments, confidences, positions = parse_pyabsa_result(prediction)
            
            # Debug: Print the prediction object for the first few examples
            if i < 3:
                print(f"\n--- Example {i+1} ---")
                print(f"Text: {text[:100]}...")
                print(f"Prediction object type: {type(prediction)}")
                print(f"Prediction: {prediction}")
                print(f"Parsed aspects: {aspects}")
                print(f"Parsed sentiments: {sentiments}")
                print(f"Parsed confidences: {confidences}")
            
            results.append({
                'text_id': i,
                'original_text': text,
                'aspects': aspects,
                'sentiments': sentiments,
                'confidences': confidences,
                'positions': positions,
                'success': True
            })
        except Exception as e:
            print(f"Error processing text {i}: {e}")
            results.append({
                'text_id': i,
                'original_text': text,
                'aspects': [],
                'sentiments': [],
                'confidences': [],
                'positions': [],
                'success': False,
                'error': str(e)
            })

    results_df = pd.DataFrame(results)
    print(f" Custom PyABSA extraction completed for {len(results_df)} samples.")

    # Save results
    output_file = processed_data_dir / 'custom_aspect_extraction.csv'
    results_df.to_csv(output_file, index=False)
    print(f" Results saved to: {output_file}")

    # Analyze results
    print(f"\n Extraction Results:")
    print(f"Success Rate: {results_df['success'].mean():.2%}")
    
    successful_df = results_df[results_df['success']]
    if len(successful_df) > 0:
        # Count total aspects extracted
        total_aspects = sum(len(aspects) for aspects in successful_df['aspects'])
        print(f"Total aspects extracted: {total_aspects}")
        
        # Get top aspects
        all_aspects = [a for sublist in successful_df['aspects'] for a in sublist]
        if all_aspects:
            top_aspects = pd.Series(all_aspects).value_counts().head(5).to_dict()
            print(f"Top 5 extracted aspects: {top_aspects}")
        
        # Get sentiment distribution
        all_sentiments = [s for sublist in successful_df['sentiments'] for s in sublist]
        if all_sentiments:
            sentiment_dist = pd.Series(all_sentiments).value_counts().to_dict()
            print(f"Sentiment distribution: {sentiment_dist}")
    
    # Show failed extractions
    failed_df = results_df[~results_df['success']]
    if len(failed_df) > 0:
        print(f"\n Failed extractions: {len(failed_df)}")
        for _, row in failed_df.head(2).iterrows():
            print(f"  - Error: {row['error'][:100]}...")

    # Show some example extractions
    print("\n Example Extractions:")
    for i, row in successful_df.head(3).iterrows():
        print(f"\nText: {row['original_text'][:100]}...")
        if row['aspects']:
            for j, (aspect, sentiment, confidence) in enumerate(zip(row['aspects'], row['sentiments'], row['confidences'])):
                print(f"  Aspect {j+1}: '{aspect}' - Sentiment: {sentiment} (Confidence: {confidence:.2f})")
        else:
            print("  No aspects extracted")

    return results_df

def main():
    try:
        results_df = run_custom_pyabsa()
        print("\n Custom PyABSA analysis completed successfully.")
    except Exception as e:
        print(f" Error during custom PyABSA analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()