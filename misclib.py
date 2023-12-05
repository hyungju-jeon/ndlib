import time


class Timer(object):
    """
    Timer context manager. It is used to measure running time during testing.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print(f"elapsed time: {self.msecs:%f} ms")
