import face_recognition # type: ignore
import os
import pickle

known_faces_dir = "known_face"
encodings_file = "known_face.dat"

known_face_encodings = []
known_face_names =[]

# loop through known face and encode them
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
        encoding = face_recognition.face_encoding(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# save encoding to a file
with open(encodings_file, "wb") as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Face encodings saved to", encodings_file)