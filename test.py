import time


start_time = time.time()
interval_seconds = 2
face_names = []
# your script
while True:
    elapsed_time = time.time() - start_time
    if elapsed_time > interval_seconds:
        # Do something
        # Reset timer
        start_time = time.time()