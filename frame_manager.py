import numpy as np

class Frames():
    def __init__(self, stack_size, frame):
        frame = frame/255.
        self.stack_size = stack_size
        if len(frame.shape) < 3:
            frame = np.expand_dims(frame,axis=2)
        frame_stack = np.repeat(frame,stack_size,axis=2)
        self.frame_stack = frame_stack

    def add_new_frame(self, frame):
        frame = frame/255.
        for i in range(1,self.stack_size):
            self.frame_stack[:,:,i-1] = self.frame_stack[:,:,i]
        self.frame_stack[:,:,self.stack_size-1] = frame

    def get_stacked_frames(self, frame=None):
        if frame is not None:
            frame = frame/255.
            self.add_new_frame(frame)
        return self.frame_stack
