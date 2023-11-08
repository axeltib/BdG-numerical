import time

def get_current_time_in_ms() -> int:
    # time() yields a result in seconds, must convert to milliseconds.
    return time.time() * 1000

