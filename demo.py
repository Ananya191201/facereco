from flask import Flask, request, jsonify
import face_recognition
import os

app = Flask(__name__)

# Load dataset
dataset_path = "dataset_path/"
known_face_encodings = []
known_face_names = []
for img_name in os.listdir(dataset_path):
    img = face_recognition.load_image_file(dataset_path + img_name)
    face_encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(img_name.split(".")[0])

# Attendance list
attendance_list = []

# Recognition route
@app.route('/recognition', methods=['POST'])
def recognition():
    # Get image from request
    img = request.files['image']
    img = face_recognition.load_image_file(img)
    
    # Recognize faces in image
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    
    recognized_names = []
    # Compare each face in the image with known faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        recognized_names.append(name)
        attendance_list.append(name)

    # Return recognized names as response
    return jsonify({"recognized_names": recognized_names})

# Attendance route
@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    global attendance_list

    # Mark attendance
    if request.method == 'POST':
        data = request.json
        name = data['name']
        attendance_list.append(name)
        return jsonify({"status": "success", "name": name})

    # View attendance
    elif request.method == 'GET':
        return jsonify({"attendance_list": attendance_list})

if __name__ == '__main__':
    app.run(debug=True)
