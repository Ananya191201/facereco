import os
import cv2
from flask import Flask, jsonify, request
import numpy as np
import face_recognition

app = Flask(__name__)
attendance = {}
# Load dataset and train the face recognition model
def train_model(dataset_path):
    dataset = {}
    for image_file in os.listdir(dataset_path):
        name = os.path.splitext(image_file)[0]
        image_path = os.path.join(dataset_path, image_file)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        dataset[name] = face_encoding
    return dataset

dataset_path = "dataset_path/"
dataset = train_model(dataset_path)



# Endpoint to add a new face to the attendance list
@app.route('/add_face', methods=['POST'])
def add_face():
    # Get the name and image from the request
    name = request.form.get('name')
    image_file = request.files['image']
    image = face_recognition.load_image_file(image_file)
    
    # Encode the face and add to the dataset
    face_encoding = face_recognition.face_encodings(image)[0]
    dataset[name] = face_encoding
    attendance[name] = False
    
    return jsonify({'message': f'{name} added to attendance list'})

# Endpoint to recognize faces and mark attendance
@app.route('/recognize', methods=['POST'])
def recognize_faces():
    # check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    # get file from post request
    image_file = request.files['image']
    
    # convert image file to numpy array
    image = np.asarray(bytearray(image_file.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # detect faces in the image
    face_locations = face_recognition.face_locations(image)
    
    # recognize faces in the image
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # get names of people recognized in the image
    recognized_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(list(dataset.values()), face_encoding, tolerance=0.6)
        name = "Unknown"
        if True in matches:
            index = matches.index(True)
            name = list(dataset.keys())[index]
        recognized_names.append(name)

    # add recognized names to attendance record
    return (recognized_names )

def update_attendance(attendance, recognized_names):
    
    for name in recognized_names:
        if name not in attendance:
            attendance.append(name)
    return attendance
    


# Endpoint to return attendance list
@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    
    return jsonify({"attendance": attendance})


if __name__ == '__main__':
    app.run(debug=True)
