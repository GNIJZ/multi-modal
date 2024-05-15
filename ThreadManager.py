import threading
import time


class ThreadManager(object):
    def __init__(self, function):
        self.thread = function


    def control_thread(self, flag):
        if flag:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if self.thread is None or not self.thread.is_alive():

            self.thread.start()

    def stop_recording(self):
        if self.thread and self.thread.is_alive():
            self.thread.stop_recording()
            self.thread.join()
