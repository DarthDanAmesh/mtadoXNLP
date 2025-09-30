
WARNING:tensorflow:From C:\Users\noob\AppData\Roaming\Python\Python313\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

Found 184834 non-empty lines in train.dat.atepc
Found 8837 non-empty lines in valid.dat.atepc
Found 9605 non-empty lines in test.dat.atepc
Training custom cybersecurity ATEPC model...

Result keys: dict_keys(['sentence', 'IOB', 'tokens', 'aspect', 'position', 'sentiment', 'probs', 'confidence'])
Text: Firewall vulnerabilities allowed unauthorized access to the network.
Result keys: dict_keys(['sentence', 'IOB', 'tokens', 'aspect', 'position', 'sentiment', 'probs', 'confidence'])
Full result: {'sentence': 'Firewall vulnerabilities allowed unauthorized access to the network .', 'IOB': ['B-ASP', 'I-ASP', 'O', 'O', 'O', 'O', 'O', 'B-ASP', 'O'], 'tokens': ['Firewall', 'vulnerabilities', 'allowed', 'unauthorized', 'access', 'to', 'the', 'network', '.'], 'aspect': ['Firewall vulnerabilities', 'network'], 'position': [[0, 1], [7]], 'sentiment': ['0', '0'], 'probs': [[0.027472639456391335, 0.9612939953804016, 0.011233419179916382], [0.03729528933763504, 0.9539980888366699, 0.00870662834495306]], 'confidence': [0.9613, 0.954]}
Aspects: ['Firewall vulnerabilities', 'network']
Polarities: []


Evaluation Results:
Total examples processed: 134
Total aspects extracted: 160
Average aspects per example: 1.57
Errors encountered: 32

Sentiment Distribution:
  Negative (-1): 51 (31.9%)
  Neutral (0): 105 (65.6%)
  Positive (1): 4 (2.5%)

Detailed results saved to: c:\Users\noob\Documents\gitProjects\mtadoXNLP\cybersecurity_absa\evaluation_results.json

Evaluation Results Analysis:
Overall Performance:
Total examples processed: 134
Total aspects extracted: 160
Average aspects per example: 1.57
Errors encountered: 32 (23.9% of examples)
Sentiment Distribution:
Negative (-1): 51 (31.9%)
Neutral (0): 105 (65.6%)
Positive (1): 4 (2.5%)
Key Insights:
Error Rate: The 32 errors (23.9% of examples) is relatively high. This could be due to:
Very long sentences (like the example shown) that might exceed the model's maximum sequence length
Complex sentence structures that the model struggles with
Unusual tokens or characters in the text
Sentiment Distribution: The model identifies mostly neutral (65.6%) and negative (31.9%) sentiments, with very few positive (2.5%) aspects. This makes sense in a cybersecurity context where:
Many aspects are described factually (neutral)
Security incidents are inherently negative
Positive aspects are rare in cybersecurity reports


model has been trained successfully with these performance metrics:

APC (Aspect Polarity Classification) Accuracy: 61.72% (max: 65.23%)
APC F1 Score: 39.93% (max: 41.24%)
ATE (Aspect Term Extraction) F1 Score: 91.44% (max: 92.59%)

These results show that your model is quite good at extracting aspect terms (91.44% F1) but has room for improvement in sentiment classification (39.93% F1). This is common in cybersecurity datasets where sentiment can be more nuanced and context-dependent.

second run of the eval:
Evaluation Results Analysis
Performance Metrics:
APC F1 Score: 41.24% (Aspect-based Polarity Classification)
ATE F1 Score: 90.57% (Aspect Term Extraction)
APC Accuracy: 65.23%
Key Observations:
Aspect Term Extraction (ATE) is strong: With an F1 score of 90.57%, your model is very good at identifying cybersecurity-related aspects in text.
Polarity Classification (PC) is moderate: With an F1 score of 41.24%, the sentiment classification is decent but has room for improvement.
Error Rate: 32 out of 134 examples (23.9%) had encoding errors

-----

For any specific domain other than what pyabsa was trained on, you can create a custom training dataset for your domain.

'bert-base-uncased'  # Using a general-purpose BERT model was used:

run, in that case:

python src/create_cybersecurity_atepc_dataset.py #first to create the dataset.
 the script does the following:
 1. Extract cybersecurity aspects from your text using pattern matching
 2. Assign sentiments based on surrounding context
 3. Create train, validation, and test splits in the PyABSA format
 4. Save them to data/custom_cybersecurity_atepc/

the expected output:
create_cybersecurity_atepc_dataset.py"
Creating custom cybersecurity ATEPC dataset...
Created train dataset with 700 samples: c:\Users\noob\Documents\gitProjects\mtadoXNLP\cybersecurity_absa\data\custom_cybersecurity_atepc\train.dat.apc
Created valid dataset with 150 samples: c:\Users\noob\Documents\gitProjects\mtadoXNLP\cybersecurity_absa\data\custom_cybersecurity_atepc\valid.dat.apc
Created test dataset with 150 samples: c:\Users\noob\Documents\gitProjects\mtadoXNLP\cybersecurity_absa\data\custom_cybersecurity_atepc\test.dat.apc

 then run
 python src/train_custom_cybersecurity_atepc.py

This will:
Load the custom cybersecurity dataset
Train a PyABSA model specifically for cybersecurity text
Save the model checkpoint to the models directory 


https://github.com/yangheng95/PyABSA/blob/v2/examples-v2/aspect_term_extraction/Aspect_Term_Extraction.ipynb
https://github.com/yangheng95/PyABSA/blob/v2/examples-v2/aspect_polarity_classification/Aspect_Sentiment_Classification.ipynb
Even though extraction ran, no aspects or sentiments were extracted from your cybersecurity text. This is due to domain mismatch.
Solution> Train a Custom ATEPC Model for Cybersecurity Domain

This is what was done to>

Created a dataset. 

Each line:
<text> ||| <aspect> ||| <sentiment>


When running with multiple aspect sentiment pairs. PyABSA struggles with the multiple aspect-sentiment pairs on a single line.

for example: if you use Multiple aspects such that>
The chinese state-sponsored [BRICKSTORM] malware is highly evasive. ||| BRICKSTORM ||| negative (single aspect)
Hackers targeted [Linux] systems for [intellectual property] theft. ||| Linux ||| neutral ||| intellectual property ||| negative (multiple aspects per line)

it will struggle.



💡 PyABSA’s pretrained ATEPC models were trained on domains like:

Restaurants

Laptops

Twitter

These models are not tuned to extract technical terms like "Linux", "APT", "persistent access", etc.

✅ Fix:
To improve extraction success:

Train a custom ATEPC model on your cybersecurity dataset (details below).
(Optional) Pre-process your text to simplify sentence structures.


[ ]Improve the preprocess_data.py (async, or any other way to improve the performance)
[ ]Improve the collect_cisa_trafilatura.py, collect_eurepoc.py, collect_csis_trafilatura.py...  (async, or any other way to improve the performance)

Based on the below running on cpu: python run_project.py
Starting Cybersecurity ABSA Project Execution...
==================================================
Working directory: c:\Users\danie\Downloads\MTXNLP\cybersecurity_absa

Running collect_eurepoc.py...
collect_eurepoc.py completed successfully

Running collect_cisa_trafilatura.py...
collect_cisa_trafilatura.py completed successfully

Running collect_csis_trafilatura.py...
collect_csis_trafilatura.py completed successfully

Running preprocess_data.py...
preprocess_data.py completed successfully

Running run_bertopic.py...
run_bertopic.py completed successfully

Running run_pyabsa_baseline.py...
run_pyabsa_baseline.py completed successfully

Running phase1_report.py...
phase1_report.py completed successfully

==================================================
Project execution completed!

[Done] exited with code=0 in 192.859 seconds