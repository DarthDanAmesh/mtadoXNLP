# src/run_bertopic.py
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import configparser
import pandas as pd
from pathlib import Path

def run_bertopic_analysis():
    # Read configuration with correct path
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent.parent / 'config.ini'
    config.read(config_path)
    
    # Get paths from config
    processed_data_dir = Path(config['paths']['processed_data_dir'])
    models_dir = Path(config['paths']['models_dir'])
    visualizations_dir = models_dir / "visualizations"
    
    # Ensure directories exist
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    
    # Load processed data
    combined_df = pd.read_csv(processed_data_dir / 'combined_dataset_phase1.csv')

    # Prepare text data for BERTopic
    texts = combined_df['clean_text'].tolist()

    # Initialize embedding model
    embedding_model = SentenceTransformer(config['topic_modeling']['embedding_model'])

    # Handle nr_topics parameter (can be integer or "auto")
    nr_topics = config['topic_modeling']['nr_topics']
    if nr_topics.isdigit():
        nr_topics = int(nr_topics)
    # else keep as "auto" string for BERTopic

    # Initialize BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=int(config['topic_modeling']['min_topic_size']),
        nr_topics=nr_topics,
        verbose=config['topic_modeling'].getboolean('verbose'),
        vectorizer_model=CountVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            min_df=2
        )
    )

    # Fit the model
    print("Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(texts)

    # Get topic information
    topic_info = topic_model.get_topic_info()
    print("Discovered Topics (BERTopic):")
    print(topic_info)

    # Add topics to the dataframe
    combined_df['bertopic_id'] = topics
    combined_df['bertopic_probability'] = probs

    # Visualize topics
    topic_visualization = topic_model.visualize_topics()
    topic_visualization.write_html(str(visualizations_dir / "bertopic_topics_visualization.html"))

    hierarchy_visualization = topic_model.visualize_hierarchy()
    hierarchy_visualization.write_html(str(visualizations_dir / "bertopic_hierarchy_visualization.html"))

    similarity_visualization = topic_model.visualize_heatmap()
    similarity_visualization.write_html(str(visualizations_dir / "bertopic_similarity_visualization.html"))

    print("BERTopic visualizations saved to models/visualizations/")

    # Save updated dataframe with topics
    combined_df.to_csv(processed_data_dir / 'dataset_with_bertopics.csv', index=False)

    # Save the model itself (optional, for later use)
    topic_model.save(str(models_dir / "bertopic_model"))

    return topic_model, combined_df, topics, probs

def main():
    try:
        topic_model, df_with_topics, topics, probs = run_bertopic_analysis()
        print(f"BERTopic modeling completed. Topics assigned to {len(df_with_topics)} records.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the data preprocessing script first.")
    except Exception as e:
        print(f"Error during BERTopic analysis: {e}")

if __name__ == "__main__":
    main()