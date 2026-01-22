import threading
import msvcrt
import queue
import time
import sys

class KeyboardListener:
    """
    A non-blocking keyboard listener that runs in a separate thread.
    Use get_key() to retrieve the latest key press.
    """
    def __init__(self):
        self.input_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def _listen(self):
        while self.running:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                self.input_queue.put(key)
            time.sleep(0.01) # Prevent high CPU usage

    def get_key(self):
        """Returns the next key from the queue, or None if empty."""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.thread.join()

class ConsolePrinter:
    """
    Helper for updating console output in place.
    """
    @staticmethod
    def print_status(message):
        sys.stdout.write(f"\r{message}")
        sys.stdout.flush()

    @staticmethod
    def clear_line():
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()
