# src/api.py
from flask import Flask, request, jsonify
from cybersecurity_atepc_inference import CybersecurityATEPC

app = Flask(__name__)
model = CybersecurityATEPC()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    result = model.analyze_text(text)
    return jsonify(result)

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'No texts provided'}), 400
    
    texts = data['texts']
    results = model.batch_analyze(texts)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)