import os
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Load the OCR model once
ocr = PaddleOCR(use_angle_cls=True, lang='en')

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    f = request.files['image']
    img = Image.open(f.stream).convert('RGB')
    img_array = np.array(img)

    result = ocr.predict(img_array)
    text_lines = [line[1][0] for line in result[0]]
    return jsonify({"text": text_lines})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
