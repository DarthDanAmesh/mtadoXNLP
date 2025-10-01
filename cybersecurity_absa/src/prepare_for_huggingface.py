# src/upload_to_huggingface.py
from huggingface_hub import HfApi, HfFolder
from pathlib import Path
import os
import shutil
import json
import re

def get_apc_f1_from_name(dirname):
    """Extract APC F1 score from directory name"""
    match = re.search(r'apcf1_(\d+\.\d+)', dirname)
    if match:
        return float(match.group(1))
    return 0.0

def prepare_and_upload_to_huggingface():
    # Set your Hugging Face token
    token = os.getenv("HUGGINGFACE_TOKEN") # Replace with your actual token
    
    # Login to Hugging Face
    HfFolder.save_token(token)
    
    # Initialize the API
    api = HfApi()
    
    # Find the best checkpoint directory
    project_root = Path(__file__).parent.parent.parent
    checkpoints_dir = project_root / 'checkpoints'
    
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
    
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"APC F1 score: {apc_f1}")
    
    # Create a temporary directory for the Hugging Face model
    temp_model_dir = project_root / 'temp_huggingface_model'
    
    # Remove the directory if it already exists
    if temp_model_dir.exists():
        shutil.rmtree(temp_model_dir)
    
    # Create the directory
    temp_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from the best checkpoint to the temporary directory
    for item in best_checkpoint.glob('*'):
        if item.is_file():
            shutil.copy2(item, temp_model_dir / item.name)
        elif item.is_dir():
            shutil.copytree(item, temp_model_dir / item.name)
    
    # Create a README.md file
    with open(temp_model_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(
            f"# Cybersecurity Aspect Term Extraction and Polarity Classification\n\n## Model Description\nThis model is trained to extract aspect terms and classify their sentiment polarity in cybersecurity texts.\n\n## Model Architecture\n- **Base Model**: BERT-base-uncased\n- **Architecture**: FAST_LCF_ATEPC\n\n## Performance Metrics\n- **APC F1**: {apc_f1}%\n- **ATE F1**: 90.57%\n- **APC Accuracy**: 65.23%\n\n## Usage\n```python\nfrom pyabsa import AspectTermExtraction as ATEPC\n\n# Load the model\naspect_extractor = ATEPC.AspectExtractor(checkpoint=\"adoamesh/PyABSA_Cybersecurity_ATE_Polarity_Classification\")\n\n# Predict aspects and sentiments\nresult = aspect_extractor.predict(\"A ransomware attack targeted the hospital's patient records system.\")\nprint(result)\n```\n\n## Training Data\nThe model was trained on a custom cybersecurity dataset with IOB format annotations.\n\n## Limitations\nThis model is trained on cybersecurity texts and may not perform well on other domains.\n\n## Biases\nThe model may reflect biases present in the training data.\n\n## License\nMIT\n\n## Author\nAdo Amesh (adoamesh@example.com)")
    
    # Model repository name
    repo_id = "adoamesh/PyABSA_Cybersecurity_ATE_Polarity_Classification"
    
    # Create the repository
    print(f"Creating repository: {repo_id}")
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    
    # Upload all files
    print("Uploading files...")
    for file_path in temp_model_dir.glob("**/*"):
        if file_path.is_file():
            # Create the path in the repository
            repo_path = str(file_path.relative_to(temp_model_dir))
            
            # Upload the file
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"Uploaded {repo_path}")
    
    # Clean up temporary directory
    shutil.rmtree(temp_model_dir)
    
    print(f"\nModel uploaded successfully to: https://huggingface.co/{repo_id}")
    print("You can now use the model with PyABSA by loading it with:")
    print("from pyabsa import AspectTermExtraction as ATEPC")
    print("aspect_extractor = ATEPC.AspectExtractor(checkpoint=\"adoamesh/PyABSA_Cybersecurity_ATE_Polarity_Classification\")")

if __name__ == "__main__":
    prepare_and_upload_to_huggingface()