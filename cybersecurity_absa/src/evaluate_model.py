# src/evaluate_model.py
from pyabsa import AspectTermExtraction as ATEPC
from pathlib import Path
import os
import re
import sys
import io

# Set stdout to handle UTF-8 properly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_apc_f1_from_name(dirname):
    """Extract APC F1 score from directory name"""
    match = re.search(r'apcf1_(\d+\.\d+)', dirname)
    if match:
        return float(match.group(1))
    return 0.0

def parse_iob_file(file_path):
    """Parse IOB format file and extract sentences"""
    sentences = []
    current_sentence = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                    current_sentence = []
            else:
                # Each line contains token and IOB tag, separated by space
                parts = line.split()
                if len(parts) >= 1:
                    current_sentence.append(parts[0])
        
        # Add the last sentence if file doesn't end with empty line
        if current_sentence:
            sentences.append(' '.join(current_sentence))
    
    return sentences

def evaluate_model():
    # Find the best checkpoint directory based on APC F1 score
    project_root = Path(__file__).parent.parent.parent
    checkpoints_dir = project_root / 'checkpoints'
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found at {checkpoints_dir}")
    
    # Get all checkpoint directories that start with "fast_lcf_atepc_custom_dataset"
    checkpoint_dirs = [
        d for d in checkpoints_dir.iterdir() 
        if d.is_dir() and d.name.startswith("fast_lcf_atepc_custom_dataset")
    ]
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {checkpoints_dir}")
    
    # Sort by APC F1 score (higher is better)
    checkpoint_dirs.sort(key=lambda d: get_apc_f1_from_name(d.name), reverse=True)
    best_checkpoint = checkpoint_dirs[0]
    apc_f1 = get_apc_f1_from_name(best_checkpoint.name)
    
    print(f"Using best checkpoint: {best_checkpoint}")
    print(f"APC F1 score: {apc_f1}")
    
    # Load the model for inference
    try:
        aspect_extractor = ATEPC.AspectExtractor(checkpoint=str(best_checkpoint))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test dataset
    test_dataset_path = Path(__file__).parent.parent / 'data' / 'custom_cybersecurity_atepc' / 'test.dat.atepc'
    
    if not test_dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_dataset_path}")
    
    print(f"Loading test dataset from: {test_dataset_path}")
    
    # Parse the IOB file to get sentences
    sentences = parse_iob_file(test_dataset_path)
    print(f"Found {len(sentences)} sentences in test dataset")
    
    # Process in batches to avoid memory issues
    batch_size = 32
    all_results = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(sentences)-1)//batch_size + 1}")
        
        try:
            batch_results = aspect_extractor.batch_predict(
                batch,
                save_result=False,
                ignore_error=True
            )
            all_results.extend(batch_results)
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add empty results for failed batch
            all_results.extend([{"error": str(e)}] * len(batch))
    
    # Process results
    total_examples = len(all_results)
    total_aspects = 0
    sentiment_distribution = {'-1': 0, '0': 0, '1': 0}
    error_count = 0
    
    for result in all_results:
        if 'error' in result:
            error_count += 1
            continue
            
        aspects = result.get('aspect', [])
        sentiments = result.get('sentiment', [])
        
        total_aspects += len(aspects)
        
        for sentiment in sentiments:
            if sentiment in sentiment_distribution:
                sentiment_distribution[sentiment] += 1
    
    print("\nEvaluation Results:")
    print(f"Total examples processed: {total_examples}")
    print(f"Total aspects extracted: {total_aspects}")
    print(f"Average aspects per example: {total_aspects / max(1, total_examples - error_count):.2f}")
    print(f"Errors encountered: {error_count}")
    print("\nSentiment Distribution:")
    print(f"  Negative (-1): {sentiment_distribution['-1']} ({sentiment_distribution['-1'] / max(1, total_aspects) * 100:.1f}%)")
    print(f"  Neutral (0): {sentiment_distribution['0']} ({sentiment_distribution['0'] / max(1, total_aspects) * 100:.1f}%)")
    print(f"  Positive (1): {sentiment_distribution['1']} ({sentiment_distribution['1'] / max(1, total_aspects) * 100:.1f}%)")
    
    # Save results to a file with UTF-8 encoding
    output_path = Path(__file__).parent.parent / 'evaluation_results_fixed.json'
    import json
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': str(best_checkpoint),
            'apc_f1': apc_f1,
            'total_examples': total_examples,
            'total_aspects': total_aspects,
            'average_aspects_per_example': total_aspects / max(1, total_examples - error_count),
            'error_count': error_count,
            'sentiment_distribution': sentiment_distribution,
            'results': all_results
        }, f, indent=2, ensure_ascii=False)  # ensure_ascii=False to properly handle Unicode
    
    print(f"\nDetailed results saved to: {output_path}")
    
    # Show some example results
    print("\nExample Results:")
    for i, result in enumerate(all_results[:5]):
        if 'error' not in result:
            print(f"\nExample {i+1}: {result['sentence']}")
            aspects = result.get('aspect', [])
            sentiments = result.get('sentiment', [])
            confidences = result.get('confidence', [])
            
            for j, aspect in enumerate(aspects):
                sentiment = sentiments[j] if j < len(sentiments) else "N/A"
                confidence = confidences[j] if j < len(confidences) else "N/A"
                print(f"  - {aspect}: {sentiment} (Confidence: {confidence})")
    
if __name__ == "__main__":
    evaluate_model()