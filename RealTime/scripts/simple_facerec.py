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

    def load_encoding_images(self, emps):
        print(f"{len(emps)} encoding images found.")
        for _, emp in emps.items():
            for imgLoc in emp["images"]:
                print(imgLoc)
                img = cv2.imread(imgLoc)
                if img is None:
                    print("Could not read image:", imgLoc)
                    continue

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                encodings = face_recognition.face_encodings(rgb_img)
                if len(encodings) > 0:
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(emp["empId"])
                else:
                    print("No face found in:", imgLoc)
        # print(self.known_face_encodings, self.known_face_names)
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
