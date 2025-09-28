# src/run_project.py
import subprocess
import sys
from pathlib import Path
import os

def run_script(script_name):
    """Run a Python script and return success status"""
    try:
        # Change to the project root directory (cybersecurity_absa)
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        
        script_path = Path("src") / script_name
        
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"{script_name} completed successfully")
            return True
        else:
            print(f"{script_name} failed with error:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    scripts = [
        "collect_eurepoc.py",
        "collect_cisa_trafilatura.py", 
        "collect_csis_trafilatura.py",
        "preprocess_data.py",
        "run_bertopic.py",
        "run_pyabsa_baseline.py",
        "phase1_report.py"
    ]
    
    print("Starting Cybersecurity ABSA Project Execution...")
    print("=" * 50)
    
    # Change to project root at the start
    project_root = Path(__file__).parent.parent
    original_cwd = os.getcwd()
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    for script in scripts:
        print(f"\nRunning {script}...")
        success = run_script(script)
        if not success:
            print(f"Stopping execution due to failure in {script}")
            break
    
    # Change back to original directory
    os.chdir(original_cwd)
    
    print("\n" + "=" * 50)
    print("Project execution completed!")

if __name__ == "__main__":
    main()