Custom Cybersecurity ATEPC Dataset (IOB Format)
===============================================

This dataset contains cybersecurity texts annotated with aspect terms and sentiments using the IOB (Inside-Outside-Begin) tagging scheme.

Format:
Each line contains: 'token IOB_tag sentiment_label'
- token: A word from the original text.
- IOB_tag: 'O' (Outside aspect), 'B-ASP' (Begin aspect), or 'I-ASP' (Inside aspect).
- sentiment_label: '1' (Positive), '0' (Neutral), or '-1' (Negative) associated with the aspect the token belongs to (or '0' if O tag).
Sentences are separated by blank lines.

Total original texts processed: 1000
Train samples (sentences): 700
Validation samples (sentences): 150
Test samples (sentences): 150
