# src/cybersecurity_atepc_inference.py
from pyabsa import AspectTermExtraction as ATEPC
from pathlib import Path
import os
import json

class CybersecurityATEPC:
    def __init__(self, checkpoint_path=None):
        """
        Initialize the Cybersecurity Aspect Term Extraction and Polarity Classification model.
        
        Args:
            checkpoint_path (str, optional): Path to the model checkpoint. 
                                           If None, uses the latest checkpoint.
        """
        self.aspect_extractor = None
        self.sentiment_map = {'-1': 'Negative', '0': 'Neutral', '1': 'Positive'}
        
        if checkpoint_path is None:
            # Find the latest checkpoint directory
            project_root = Path(__file__).parent.parent.parent
            checkpoints_dir = project_root / 'checkpoints'
            
            if not checkpoints_dir.exists():
                raise FileNotFoundError(f"Checkpoints directory not found at {checkpoints_dir}")
            
            # Get all checkpoint directories and find the most recent
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
            if not checkpoint_dirs:
                raise FileNotFoundError(f"No checkpoint directories found in {checkpoints_dir}")
            
            checkpoint_path = str(max(checkpoint_dirs, key=os.path.getmtime))
            print(f"Using checkpoint: {checkpoint_path}")
        
        # Load the model
        try:
            self.aspect_extractor = ATEPC.AspectExtractor(checkpoint=checkpoint_path)
            print("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def analyze_text(self, text):
        """
        Analyze a text to extract aspects and their sentiment polarities.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            dict: A dictionary containing the aspects, sentiments, and confidences.
        """
        if self.aspect_extractor is None:
            raise RuntimeError("Model not loaded. Call __init__ first.")
        
        try:
            result = self.aspect_extractor.predict(
                text,
                save_result=False,
                ignore_error=True
            )
            
            # Extract aspects, sentiments, and confidences
            aspects = result.get('aspect', [])
            sentiments = result.get('sentiment', [])
            confidences = result.get('confidence', [])
            
            # Create a list of aspect-sentiment pairs
            aspect_sentiments = []
            for i, aspect in enumerate(aspects):
                sentiment_label = self.sentiment_map.get(sentiments[i], sentiments[i])
                confidence = confidences[i] if i < len(confidences) else "N/A"
                aspect_sentiments.append({
                    "aspect": aspect,
                    "sentiment": sentiment_label,
                    "confidence": confidence
                })
            
            return {
                "text": text,
                "aspects": aspect_sentiments
            }
        except Exception as e:
            return {
                "text": text,
                "error": str(e)
            }
    
    def batch_analyze(self, texts):
        """
        Analyze a batch of texts.
        
        Args:
            texts (list): A list of texts to analyze.
            
        Returns:
            list: A list of dictionaries, each containing the analysis for one text.
        """
        return [self.analyze_text(text) for text in texts]

def main():
    # Example usage
    try:
        # Initialize the model
        model = CybersecurityATEPC()
        
        # Example texts
        test_texts = [
            "A ransomware attack targeted the hospital's patient records system.",
            "The phishing campaign compromised employee credentials.",
            "Firewall vulnerabilities allowed unauthorized access to the network."
        ]
        
        # Analyze texts
        results = model.batch_analyze(test_texts)
        
        # Print results
        for result in results:
            print(f"\nText: {result['text']}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                for aspect_data in result['aspects']:
                    print(f"  - {aspect_data['aspect']}: {aspect_data['sentiment']} (Confidence: {aspect_data['confidence']})")
        
        # Save results to JSON
        with open('cybersecurity_atepc_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to cybersecurity_atepc_results.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()