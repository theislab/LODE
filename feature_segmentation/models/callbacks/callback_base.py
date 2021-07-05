from abc import ABC


class Callback(ABC):
    def on_epoch_start(self, frame, detections):
        pass

    def on_epoch_end(self, frame, tracks, dets):
        pass


class CallbackManagement:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_epoch_start(self, frame, detections):
        for callback in self.callbacks:
            callback.on_frame_start(frame, detections)

    def on_epoch_end(self, frame, tracks, dets):
        for callback in reversed(self.callbacks):
            callback.on_frame_end(frame, tracks, dets)