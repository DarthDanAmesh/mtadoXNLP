# src/analyze_errors.py
import json
from pathlib import Path

def analyze_errors():
    # Load the evaluation results
    results_path = Path(__file__).parent.parent / 'evaluation_results.json'
    
    with open(results_path, 'r') as f:
        evaluation_data = json.load(f)
    
    results = evaluation_data['results']
    
    # Filter for examples with errors
    error_examples = []
    for i, result in enumerate(results):
        if 'error' in result:
            error_examples.append((i, result))
    
    print(f"Found {len(error_examples)} examples with errors")
    
    # Load the original test dataset to compare
    test_dataset_path = Path(__file__).parent.parent / 'data' / 'custom_cybersecurity_atepc' / 'test.dat.atepc'
    
    sentences = []
    current_sentence = []
    
    with open(test_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(' '.join(current_sentence))
                    current_sentence = []
            else:
                parts = line.split()
                if len(parts) >= 1:
                    current_sentence.append(parts[0])
        
        if current_sentence:
            sentences.append(' '.join(current_sentence))
    
    # Show error examples
    print("\nError Examples:")
    for i, (idx, error_result) in enumerate(error_examples[:5]):  # Show first 5 errors
        if idx < len(sentences):
            print(f"\nError {i+1}:")
            print(f"Sentence: {sentences[idx]}")
            print(f"Error: {error_result['error']}")
    
    # Analyze sentence length for errors vs successful examples
    error_lengths = []
    success_lengths = []
    
    for i, result in enumerate(results):
        if i < len(sentences):
            sentence_length = len(sentences[i].split())
            if 'error' in result:
                error_lengths.append(sentence_length)
            else:
                success_lengths.append(sentence_length)
    
    print("\nSentence Length Analysis:")
    print(f"Average length of sentences with errors: {sum(error_lengths) / len(error_lengths):.1f} words")
    print(f"Average length of successful sentences: {sum(success_lengths) / len(success_lengths):.1f} words")
    print(f"Maximum length of sentences with errors: {max(error_lengths)} words")
    print(f"Maximum length of successful sentences: {max(success_lengths)} words")

if __name__ == "__main__":
    analyze_errors()