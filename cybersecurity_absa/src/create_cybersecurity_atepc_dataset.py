# src/create_cybersecurity_atepc_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path
import configparser
import re
import random

def load_config():
    """Load configuration from config.ini"""
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent / 'config.ini'
    config.read(config_path)
    return config

def extract_aspects_from_text(text):
    """
    Extract potential cybersecurity aspects from text using pattern matching
    """
    # Common cybersecurity aspect patterns
    aspect_patterns = [
        r'\b(vulnerability|vulnerabilities)\b',
        r'\b(exploit|exploits|exploiting)\b',
        r'\b(malware|viruses|trojans|ransomware|spyware)\b',
        r'\b(phishing|phish)\b',
        r'\b(data breach|data leak|information leak)\b',
        r'\b(Ddos|DDoS|denial of service)\b',
        r'\b(firewall|firewalls)\b',
        r'\b(antivirus|anti-virus)\b',
        r'\b(encryption|encrypt)\b',
        r'\b(authentication|authenticating)\b',
        r'\b(authorization|authorizing)\b',
        r'\b(password|passwords)\b',
        r'\b(patch|patches|patching)\b',
        r'\b(update|updates|updating)\b',
        r'\b(backup|backups)\b',
        r'\b(network|networks)\b',
        r'\b(server|servers)\b',
        r'\b(database|databases)\b',
        r'\b(system|systems)\b',
        r'\b(security|secure)\b',
        r'\b(threat|threats)\b',
        r'\b(attack|attacks|attacking)\b',
        r'\b(defense|defenses|defending)\b',
        r'\b(protection|protecting)\b',
        r'\b(detection|detecting)\b',
        r'\b(prevention|preventing)\b',
        r'\b(response|responding)\b',
        r'\b(recovery|recovering)\b',
        r'\b(incident|incidents)\b',
        r'\b(breach|breaches)\b',
        r'\b(intrusion|intrusions)\b',
        r'\b(compromise|compromised)\b',
        r'\b(hacker|hackers|hacking)\b',
        r'\b(cyberattack|cyberattacks)\b'
    ]
    
    aspects = []
    for pattern in aspect_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            aspects.append((match.group().lower(), match.start(), match.end()))
    
    # Sort by start position and remove overlaps
    aspects = sorted(aspects, key=lambda x: x[1])
    non_overlapping_aspects = []
    for aspect in aspects:
        if not non_overlapping_aspects or non_overlapping_aspects[-1][2] <= aspect[1]:
            non_overlapping_aspects.append(aspect)
    return non_overlapping_aspects

def assign_sentiment_to_aspect(text, aspect, aspect_start, aspect_end):
    """
    Assign sentiment to an aspect based on surrounding context
    """
    # Extract a window around the aspect
    window_size = 20  # Number of words to consider on each side
    words = text.split()
    # Calculate the word indices corresponding to the character positions
    char_before_aspect = text[:aspect_start]
    words_before = char_before_aspect.split()
    start_word_idx = len(words_before)
    
    char_aspect = text[aspect_start:aspect_end]
    aspect_words = char_aspect.split()
    end_word_idx = start_word_idx + len(aspect_words)

    # Determine context words for sentiment analysis
    context_start = max(0, start_word_idx - window_size)
    context_end = min(len(words), end_word_idx + window_size)
    context = ' '.join(words[context_start:context_end])
    
    # Positive sentiment indicators
    positive_words = [
        'effective', 'robust', 'secure', 'protected', 'safe', 'strong',
        'reliable', 'successful', 'improved', 'enhanced', 'fixed', 'resolved',
        'prevented', 'blocked', 'detected', 'mitigated', 'restored'
    ]
    
    # Negative sentiment indicators
    negative_words = [
        'vulnerable', 'compromised', 'breached', 'attacked', 'failed',
        'weak', 'exploited', 'infected', 'corrupted', 'lost', 'stolen',
        'unauthorized', 'malicious', 'dangerous', 'risky', 'insecure',
        'disrupted', 'down', 'unavailable', 'crashed', 'hacked', 'encrypted',
        'inaccessible', 'standstill', 'disaster', 'failure', 'sabotaged'
    ]
    
    # Count positive and negative words in context
    positive_count = sum(1 for word in positive_words if word.lower() in context.lower())
    negative_count = sum(1 for word in negative_words if word.lower() in context.lower())
    
    # Determine sentiment
    if positive_count > negative_count:
        return 'Positive'
    elif negative_count > positive_count:
        return 'Negative'
    else:
        # Default to Neutral if counts are equal or no matches
        return 'Neutral'

def clean_text(text):
    """Clean text to ensure it's suitable for training"""
    # Remove special characters that might cause issues, but keep essential punctuation
    # text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"\/\(\)]', ' ', text)
    # For IOB format, keeping punctuation attached to words might be fine, but extra spaces are not
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_atepc_dataset(input_file, output_dir, sample_size=1000):
    """
    Create a custom ATEPC dataset from cybersecurity text using IOB format.
    """
    # Load the input data
    df = pd.read_csv(input_file)
    
    # Clean the text
    df['clean_text'] = df['clean_text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['clean_text'].str.len() > 10]
    
    # Sample texts if the dataset is large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # Map sentiment labels to numerical values expected by PyABSA ATEPC
    label_map = {'Positive': '1', 'Neutral': '0', 'Negative': '-1'}

    # Process each split
    for split_name, split_df in [('train', train_df), ('valid', val_df), ('test', test_df)]:
        output_file = output_dir / f'{split_name}.dat.atepc' # Use .atepc extension
        
        valid_samples = 0
        skipped_short_text = 0
        skipped_no_aspects = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in split_df.iterrows():
                text = row['clean_text']
                
                # Skip texts that are too short
                if len(text) < 20:
                    skipped_short_text += 1
                    continue
                    
                aspects = extract_aspects_from_text(text)
                
                if not aspects:
                    skipped_no_aspects += 1
                    continue # Skip if no aspects found
                    
                # Tokenize the text (simple split by space might be okay, or use a proper tokenizer)
                tokens = text.split() 
                # Create IOB tags for each token
                iob_tags = ['O'] * len(tokens)
                sent_labels = ['0'] * len(tokens) # Default label

                # Apply aspect tags and sentiments
                for aspect_text, start_char, end_char in aspects:
                    aspect_tokens = aspect_text.split()
                    # Find the start index of the aspect tokens in the full token list
                    found = False
                    for i in range(len(tokens)):
                        if ' '.join(tokens[i:i+len(aspect_tokens)]).lower() == aspect_text.lower():
                            # Assign B-ASP to the first token
                            iob_tags[i] = 'B-ASP'
                            # Assign I-ASP to subsequent tokens if more than one
                            for j in range(1, len(aspect_tokens)):
                                if i+j < len(iob_tags):
                                    iob_tags[i+j] = 'I-ASP'
                            
                            # Determine sentiment for the whole aspect span
                            sentiment_str = assign_sentiment_to_aspect(text, aspect_text, start_char, end_char)
                            sentiment_num = label_map.get(sentiment_str, '0') # Default to Neutral if not found

                            # Assign the determined sentiment to all tokens in the aspect span
                            for j in range(len(aspect_tokens)):
                                if i+j < len(sent_labels):
                                    sent_labels[i+j] = sentiment_num
                            
                            found = True
                            break # Move to the next aspect
                    # if not found: # Optional: Log if an aspect couldn't be matched to tokens
                    #     print(f"Warning: Could not align aspect '{aspect_text}' in text: {text}")

                # Write the IOB format lines
                for token, tag, label in zip(tokens, iob_tags, sent_labels):
                    f.write(f"{token} {tag} {label}\n")
                
                # Add a blank line to separate sentences/documents
                f.write("\n")
                
                valid_samples += 1 # Count sentences, not individual aspect occurrences


        print(f"Created {split_name} dataset with {valid_samples} valid samples (sentences): {output_file}")
        print(f"  - Skipped {skipped_short_text} short/problematic texts")
        print(f"  - Skipped {skipped_no_aspects} texts with no aspects (that produced no final lines)")
    
    # Create a readme file for the dataset
    readme_file = output_dir / 'readme.txt'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("Custom Cybersecurity ATEPC Dataset (IOB Format)\n")
        f.write("===============================================\n\n")
        f.write("This dataset contains cybersecurity texts annotated with aspect terms and sentiments using the IOB (Inside-Outside-Begin) tagging scheme.\n\n")
        f.write("Format:\n")
        f.write("Each line contains: 'token IOB_tag sentiment_label'\n")
        f.write("- token: A word from the original text.\n")
        f.write("- IOB_tag: 'O' (Outside aspect), 'B-ASP' (Begin aspect), or 'I-ASP' (Inside aspect).\n")
        f.write("- sentiment_label: '1' (Positive), '0' (Neutral), or '-1' (Negative) associated with the aspect the token belongs to (or '0' if O tag).\n")
        f.write("Sentences are separated by blank lines.\n\n")
        f.write(f"Total original texts processed: {len(df)}\n")
        f.write(f"Train samples (sentences): {len(train_df)}\n")
        f.write(f"Validation samples (sentences): {len(val_df)}\n")
        f.write(f"Test samples (sentences): {len(test_df)}\n")
    
    print(f"Created readme file: {readme_file}")
    return output_dir

def main():
    # Load configuration
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent / 'config.ini'
    config.read(config_path)
    
    # Get paths
    project_root = Path(__file__).parent.parent
    processed_data_dir = project_root / config['paths']['processed_data_dir']
    custom_dataset_dir = project_root / 'data' / 'custom_cybersecurity_atepc'
    
    # Input file
    input_file = processed_data_dir / 'combined_dataset_with_topics.csv'
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create custom dataset
    print("Creating custom cybersecurity ATEPC dataset (IOB format)...")
    dataset_dir = create_atepc_dataset(input_file, custom_dataset_dir)
    
    print(f"\nCustom ATEPC dataset (IOB format) created at: {dataset_dir}")
    print("You can now use this dataset to train a custom PyABSA ATEPC model.")

if __name__ == "__main__":
    main()