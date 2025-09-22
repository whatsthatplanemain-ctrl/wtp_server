from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file.stream)

    # For now, just return a dummy response
    return jsonify({'result': 'Hello from Flask! Your image was received.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
