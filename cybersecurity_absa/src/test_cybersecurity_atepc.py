# src/test_cybersecurity_atepc.py
from pyabsa import AspectTermExtraction as ATEPC
from pathlib import Path
import os

def test_model():
    # Find the latest checkpoint directory
    project_root = Path(__file__).parent.parent.parent
    checkpoints_dir = project_root / 'checkpoints'
    
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found at {checkpoints_dir}")
    
    # Get all checkpoint directories and find the most recent
    checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {checkpoints_dir}")
    
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
    print(f"Using checkpoint: {latest_checkpoint}")
    
    # Load the model for inference
    try:
        aspect_extractor = ATEPC.AspectExtractor(checkpoint=str(latest_checkpoint))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test with example cybersecurity texts
    test_texts = [
        "A ransomware attack targeted the hospital's patient records system.",
        "The phishing campaign compromised employee credentials.",
        "Firewall vulnerabilities allowed unauthorized access to the network."
    ]
    
    for text in test_texts:
        try:
            result = aspect_extractor.predict(
                text,
                save_result=False,
                ignore_error=True
            )
            print(f"\nText: {text}")
            
            # Extract aspects and sentiments
            aspects = result.get('aspect', [])
            sentiments = result.get('sentiment', [])
            confidences = result.get('confidence', [])
            
            print(f"Aspects: {aspects}")
            print(f"Sentiments: {sentiments}")
            print(f"Confidences: {confidences}")
            
            # Map sentiment values to human-readable labels
            sentiment_map = {'-1': 'Negative', '0': 'Neutral', '1': 'Positive'}
            
            # Print each aspect with its sentiment and confidence
            for i, aspect in enumerate(aspects):
                sentiment_label = sentiment_map.get(sentiments[i], sentiments[i])
                confidence = confidences[i] if i < len(confidences) else "N/A"
                print(f"  - {aspect}: {sentiment_label} (Confidence: {confidence})")
                
        except Exception as e:
            print(f"Error predicting for text '{text}': {e}")
            import traceback
            traceback.print_exc()
        
if __name__ == "__main__":
    test_model()