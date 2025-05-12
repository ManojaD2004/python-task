import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        image_files = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(image_files)} encoding images found.")

        for img_path in image_files:
            print(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print("Could not read image:", img_path)
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            basename = os.path.basename(img_path)
            filename, _ = os.path.splitext(basename)

            encodings = face_recognition.face_encodings(rgb_img)
            if len(encodings) > 0:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(filename)
            else:
                print("No face found in:", img_path)
        print("Encoding images loaded.")

    def detect_known_faces(self, frame):
        resize_factor = self.frame_resizing
        small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding
            )
            name = "Unknown"
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations)
        if face_locations.size != 0:
            face_locations = (face_locations / resize_factor).astype(int)
        return face_locations, face_names
