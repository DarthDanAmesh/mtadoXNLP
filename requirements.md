# Core NLP and ML, Data Processing and Web Scrapping libraries

need at Microsoft Visual C++ 14.0 or greater. https://visualstudio.microsoft.com/visual-cpp-build-tools/
conta create -n pyabsa
conda activate pyabsa
pip install pyabsa trafilatura bertopic hf_xet
python -m spacy download en_core_web_lg

pip install tensorflow #or for linux systems: pip install tensorflow[and-cuda]
pip install tf-keras

Run the data collection scripts in this order:
1. python src/collect_eurepoc.py
2. python src/collect_cisa_trafilatura.py
3. python src/collect_csis_trafilatura.py
4. python src/preprocess_data.py
5. python src/run_bertopic.py
6. python src/run_pyabsa_baseline.py
7. python src/phase1_report.py