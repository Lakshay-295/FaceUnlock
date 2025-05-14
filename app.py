from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load known face
known_image = face_recognition.load_image_file("known_face.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({"result": "error", "message": "No image data"}), 400

    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    rgb_img = img[:, :, ::-1]
    encodings = face_recognition.face_encodings(rgb_img)

    if encodings:
        match = face_recognition.compare_faces([known_encoding], encodings[0])
        return jsonify({"result": "unlock" if match[0] else "deny"})
    else:
        return jsonify({"result": "deny", "message": "No face found"})

@app.route('/', methods=['GET'])
def home():
    return 'Face Recognition Server is Running'
