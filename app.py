from api import *
from timer import Timer
import face_recognition
import cv2
import numpy as np


# Config
# You MUST add the following line in the hosts file
# 64.227.36.148 htms.com

HOST_URL = 'http://htms.com'
SITE_ID = 1  # Where is the facial recognition placed, on which site
FACES_DIRECTORY = 'faces'

# Setup
api = API(host=HOST_URL)
profiles = api.get_profiles()
database = ImageDatabase(images_dir=FACES_DIRECTORY, profiles=profiles)
# Synchronise the faces from database with the ones on the disk
database.sync()

# Run the Facial recognition process
# HTMS system Facial Recognition Software
# Student ID: 19040847
# Team Project Assignment Semester B

# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

for profile in profiles:
    # Get the profile image path
    image_path = database.make_disk_path(profile.get_image_disk_name())
    # Load a sample picture and learn how to recognize it.
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    # Add the encoding and the name of the profile to the arrays
    known_face_encodings.append(encoding)
    known_face_names.append(profile.full_name)

# Initialize some variables to be used later
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Access and display the default webcam's input.
video_capture = cv2.VideoCapture(0)
# Timer
timer = Timer()
interval_seconds = 2
# we use this array to store temporary names so we can check them if they stayed in the frame for longer time
temp_names = []
while True:
    # Decrease the timer
    timer.update()
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

    # Check if the recognised face is present for an interval of seconds
    # If the same face is present for the required interface then we perform an API request to sign the employee
    process_this_frame = not process_this_frame
    if timer.has_passed(interval_seconds):
        for name in face_names:
            # It means that the face is still preset and we proceed to sign it
            if name in temp_names:
                # Find the profile by the recognised name
                profile = find_profile_by_name(name, profiles)
                # Normally the profile it should never be None, but still we will do a safe check just in case
                if profile is not None:
                    # Sign the user and get the response
                    data = api.sign(profile.user_id, SITE_ID)
                    message = data['message']
                    signsIn = bool(data['signsIn'])
                    print(message)

                # We done our job with this name, we will remove it
                temp_names.remove(name)
            else:
                # We will add it for later reference
                temp_names.append(name)

        # Reset timer
        timer.reset()

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
