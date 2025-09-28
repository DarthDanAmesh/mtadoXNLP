# src/run_pyabsa_baseline.py
from pyabsa import AspectTermExtraction as ATEPC
import configparser
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def run_pyabsa_baseline():
    # Read configuration with correct path
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent / 'config.ini'
    config.read(config_path)
    
    # Get paths from config
    processed_data_dir = Path(config['paths']['processed_data_dir'])
    
    # Load dataset with BERTopic assignments
    df_with_topics = pd.read_csv(processed_data_dir / 'dataset_with_bertopics.csv')

    # Initialize PyABSA (using config from Task 1.4)
    aspect_extractor = ATEPC.AspectExtractor(checkpoint="english") # Load a pre-trained checkpoint

    # Sample texts for baseline extraction (limit for initial run)
    sample_texts = df_with_topics['clean_text'].head(20).tolist() # Adjust sample size as needed

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
            # Process results
            aspect_data = {
                'text_id': i,
                'original_text': text,
                'aspects': prediction.get('aspect', []),
                'sentiments': prediction.get('sentiment', []),
                'confidences': prediction.get('confidence', []),
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
    baseline_results_df.to_csv(processed_data_dir / 'baseline_aspect_extraction.csv', index=False)

    # Analyze results (basic)
    print(f"Baseline Success Rate: {baseline_results_df['success'].mean():.2%}")
    successful_df = baseline_results_df[baseline_results_df['success']]
    if len(successful_df) > 0:
        all_aspects = [aspect for sublist in successful_df['aspects'] for aspect in sublist]
        unique_aspects = pd.Series(all_aspects).value_counts()
        print(f"Top 5 extracted aspects: {unique_aspects.head(5).to_dict()}")
        all_sentiments = [sentiment for sublist in successful_df['sentiments'] for sentiment in sublist]
        sentiment_dist = pd.Series(all_sentiments).value_counts()
        print(f"Sentiment distribution: {sentiment_dist.to_dict()}")

    return baseline_results_df

def main():
    try:
        baseline_results_df = run_pyabsa_baseline()
        print(f"PyABSA baseline analysis completed successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the BERTopic analysis script first.")
    except Exception as e:
        print(f"Error during PyABSA baseline analysis: {e}")

if __name__ == "__main__":
    main()