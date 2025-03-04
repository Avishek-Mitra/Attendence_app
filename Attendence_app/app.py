from flask import Flask, render_template, request, redirect, url_for
import face_recognition # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import pickle
import csv
from datetime import datetime

app = Flask(__name__)

# Load known face encodings
with open("known_faces.dat", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Attendance CSV file
attendance_file = "attendance.csv"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():
    # Capture image from webcam
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    video_capture.release()

    # Find faces in the captured frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # Compare face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            # Mark attendance
            with open(attendance_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)