# src/cybersecurity_atepc_pipeline.py
from pathlib import Path
import json
from cybersecurity_atepc_inference import CybersecurityATEPC

class CybersecurityATEPCPipeline:
    def __init__(self, checkpoint_path=None):
        """Initialize the pipeline with a trained model."""
        self.model = CybersecurityATEPC(checkpoint_path)
    
    def process_text(self, text):
        """Process a single text and return aspects with sentiments."""
        return self.model.analyze_text(text)
    
    def process_batch(self, texts):
        """Process a batch of texts."""
        return self.model.batch_analyze(texts)
    
    def process_file(self, input_file, output_file):
        """Process a text file and save results."""
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = self.process_batch(texts)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate(self, test_file):
        """Evaluate the model on a test dataset."""
        # Load the evaluation script
        from evaluate_model import evaluate_model
        return evaluate_model()

# Example usage
if __name__ == "__main__":
    pipeline = CybersecurityATEPCPipeline()
    
    # Process a single text
    result = pipeline.process_text("A ransomware attack targeted the hospital's patient records system.")
    print(result)
    
    # Process a file
    pipeline.process_file("input.txt", "output.json")
    
    # Evaluate the model
    pipeline.evaluate("test.dat.atepc")