import time


class Timer:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.elapsed_time = None

    def reset(self):
        self.start_time = time.time()

    def update(self):
        self.elapsed_time = time.time() - self.start_time

    def has_passed(self, seconds) -> bool:
        return self.elapsed_time > seconds
