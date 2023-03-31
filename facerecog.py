import face_recognition
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# load dataset
dataset_path=  'dataset_path/'
def load_dataset(dataset_path):
    dataset = {}
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            encodings = []
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                encodings.append(encoding)
            dataset[folder] = encodings
    return dataset

# mark attendance
def mark_attendance(attendance, name):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    attendance[name] = dt_string
    return attendance

# recognize faces in image
def recognize(image_path, dataset):
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, locations)

    names = []
    recognized_names = []

    for encoding in encodings:
        for name, encodings in dataset.items():
            matches = face_recognition.compare_faces(encodings, encoding)
            if True in matches:
                names.append(name)
                recognized_names.append(name)
                break
        else:
            names.append('name')

    return names, recognized_names

attendance = {}

# endpoint to mark attendance
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_endpoint():
    image_path = request.files['image_path']
    name = request.form.get['name']
    attendance = mark_attendance(attendance, name)
    response = {'message': 'Attendance marked successfully'}
    return jsonify(response)

# endpoint to recognize faces in image
@app.route('/recognize_faces', methods=['POST'])
def recognize_faces_endpoint():
    image_path = request.files['image_path']
    names, recognized_names = recognize(image_path, dataset)
    attendance_list = []
    for name in names:
        attendance_list.append({'name': name, 'status': attendance.get(name, 'Present')})
    response = {'attendance': attendance_list, 'recognized_names': recognized_names}
    return jsonify(response)

# endpoint to get attendance
@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    attendance_list = []
    for name, status in attendance.items():
        attendance_list.append({'name': name, 'status': status})
    response = {'attendance': attendance_list}
    return jsonify(response)

if __name__ == '__main__':

    dataset = load_dataset(dataset_path)
    app.run(debug=True)
