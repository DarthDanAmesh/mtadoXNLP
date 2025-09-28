# src/run_bertopic.py

# === Suppress TensorFlow verbosity and warnings ===
import warnings
import os
import logging

# Set TensorFlow environment variables to minimize logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide INFO, WARNING, and ERROR logs
os.environ["TF_DEPRECATION_WARNINGS"] = "0"  # Disable TF deprecation warnings

# Configure Python warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*tf\\.losses\\.sparse_softmax_cross_entropy.*")
warnings.filterwarnings("ignore", message=".*From tf_keras.*")

# Configure logging to suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tf_keras').setLevel(logging.ERROR)

# === Now import everything else ===
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import configparser
import pandas as pd
from pathlib import Path

def run_bertopic_analysis():
    # --- Resolve project root and config ---
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.ini'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if 'paths' not in config:
        raise ValueError("Missing '[paths]' section in config.ini")
    if 'topic_modeling' not in config:
        raise ValueError("Missing '[topic_modeling]' section in config.ini")

    # --- Resolve directories ---
    try:
        processed_data_dir = project_root / config['paths']['processed_data_dir']
        models_dir = project_root / config['paths']['models_dir']
        visualizations_dir = models_dir / "visualizations"
    except KeyError as e:
        raise KeyError(f"Missing key in config.ini: {e}")

    # Ensure output directories exist
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    # --- Load input data ---
    input_file = processed_data_dir / 'combined_dataset_phase1.csv'
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input data not found: {input_file}\n"
            "Please run: python src/preprocess_data.py first"
        )

    print(f"Loading data from: {input_file}")
    combined_df = pd.read_csv(input_file)

    if combined_df.empty:
        raise ValueError("Input dataset is empty.")

    if 'clean_text' not in combined_df.columns:
        raise KeyError(
            "'clean_text' column missing in input data. "
            "Ensure preprocessing was completed successfully."
        )

    texts = combined_df['clean_text'].dropna().tolist()
    if not texts:
        raise ValueError("No valid text data found in 'clean_text' column.")

    print(f"Processing {len(texts)} documents with BERTopic...")

    # --- Configure embedding model ---
    embedding_model_name = config['topic_modeling']['embedding_model']
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)

    # --- Parse nr_topics (support 'auto' or integer) ---
    nr_topics = config['topic_modeling']['nr_topics'].strip()
    if nr_topics.isdigit():
        nr_topics = int(nr_topics)
    elif nr_topics.lower() == 'auto':
        nr_topics = "auto"
    else:
        raise ValueError(f"Invalid 'nr_topics' in config: expected integer or 'auto', got '{nr_topics}'")

    # --- Initialize BERTopic ---
    min_topic_size = int(config['topic_modeling']['min_topic_size'])
    verbose = config['topic_modeling'].getboolean('verbose')

    print(f"Initializing BERTopic (min_topic_size={min_topic_size}, nr_topics={nr_topics}, verbose={verbose})...")
    
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=2
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        verbose=verbose,
        vectorizer_model=vectorizer_model
    )

    # --- Fit model ---
    print("Fitting BERTopic model... (this may take several minutes)")
    topics, probs = topic_model.fit_transform(texts)

    # --- Inspect topics ---
    topic_info = topic_model.get_topic_info()
    print("\nDiscovered Topics (Top 10):")
    print(topic_info.head(10))

    # --- Add topic info to dataframe ---
    combined_df = combined_df.copy()
    combined_df['bertopic_id'] = topics
    combined_df['bertopic_probability'] = probs

    # --- Save visualizations ---
    print("Generating visualizations...")
    
    # Save topic model
    model_path = models_dir / "bertopic_model"
    topic_model.save(str(model_path))
    print(f"Model saved to: {model_path}")
    
    # Save topic info
    topic_info_path = models_dir / "bertopic_topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False)
    print(f"Topic info saved to: {topic_info_path}")
    
    # Save updated dataframe
    output_path = processed_data_dir / "combined_dataset_with_topics.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"Dataset with topics saved to: {output_path}")
    
    # Generate visualizations
    try:
        # Topic visualization
        topic_viz = topic_model.visualize_topics()
        topic_viz_path = visualizations_dir / "bertopic_topics.html"
        topic_viz.write_html(str(topic_viz_path))
        print(f"Topics visualization saved to: {topic_viz_path}")
        
        # Topic hierarchy visualization
        hierarchy_viz = topic_model.visualize_hierarchy()
        hierarchy_viz_path = visualizations_dir / "bertopic_hierarchy.html"
        hierarchy_viz.write_html(str(hierarchy_viz_path))
        print(f"Topic hierarchy visualization saved to: {hierarchy_viz_path}")
        
        # Word cloud visualization
        wordcloud_viz = topic_model.visualize_barchart()
        wordcloud_viz_path = visualizations_dir / "bertopic_wordcloud.html"
        wordcloud_viz.write_html(str(wordcloud_viz_path))
        print(f"Word cloud visualization saved to: {wordcloud_viz_path}")
        
        # Topic similarity heatmap
        heatmap_viz = topic_model.visualize_heatmap()
        heatmap_viz_path = visualizations_dir / "bertopic_heatmap.html"
        heatmap_viz.write_html(str(heatmap_viz_path))
        print(f"Topic similarity heatmap saved to: {heatmap_viz_path}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    print("\nBERTopic analysis completed successfully!")
    return combined_df, topic_model

if __name__ == "__main__":
    run_bertopic_analysis()