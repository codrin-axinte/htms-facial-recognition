import face_recognition
import cv2
import numpy as np

# HTMS system Facial Recognition Software
# Student ID: 19040847
# Team Project Assignment Semester B


# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)


# Access and display the default webcam's input.
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
luke_image = face_recognition.load_image_file("faces/luke.jpg")
luke_face_encoding = face_recognition.face_encodings(luke_image)[0]

# Load a second sample picture and learn how to recognize it.
dylan_image = face_recognition.load_image_file("faces/dylan.jpg")
dylan_face_encoding = face_recognition.face_encodings(dylan_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    luke_face_encoding,
    dylan_face_encoding
]

known_face_names = [
    "luke",
    "dylan"
]

# Initialize some variables to be used later
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size to make the facial recognition faster
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (200, 100, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (200, 100, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam window and everything attaching to it.
video_capture.release()
cv2.destroyAllWindows()
