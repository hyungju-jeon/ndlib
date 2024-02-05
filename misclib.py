"""
Library fo miscellaneous functions.
"""

import time
import pyautogui
import numpy as np

CURS_X_MAX, CURS_Y_MAX = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.DARWIN_CATCH_UP_TIME = 1e-3


def get_cursor_pos():
    """
    Returns the current position of the mouse cursor.

    Returns:
        tuple: A tuple containing the x and y coordinates of the mouse cursor.
    """
    coords = pyautogui.position()
    # center and normalize
    coords = 2 * np.array(coords) / np.array([CURS_X_MAX, CURS_Y_MAX]) - 1

    return coords


def update_cursor_pos(new_pos):
    """
    Updates the position of the mouse cursor.

    Parameters:
        new_pos (tuple): A tuple containing the new x and y coordinates of the mouse cursor.

    Returns:
        None
    """
    new_pos = (new_pos + 1) * np.array([CURS_X_MAX, CURS_Y_MAX]) / 2
    # clip to screen
    new_pos = np.clip(new_pos, 0, np.array([CURS_X_MAX, CURS_Y_MAX]))
    pyautogui.moveTo(new_pos[0], new_pos[1], _pause=False)


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
