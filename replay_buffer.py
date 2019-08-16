import numpy as np

class Replay_Buffer():
    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity
        self.pos = 0
        self.replay_buffer = []

    def add_replay(self, replay):
        if len(self.replay_buffer)>=self.buffer_capacity:
            self.replay_buffer[self.pos] = replay
            self.pos = (self.pos+1) % self.buffer_capacity
        else:
            self.replay_buffer.append(replay)

    def get_sample(self,m):
        size_of_replay_buffer = len(self.replay_buffer)
        print('size_of_replay_buffer',size_of_replay_buffer)
        replay_indicies = np.random.choice(size_of_replay_buffer, m)
        replays = [self.replay_buffer[j] for j in replay_indicies]
        action = [replay[0] for replay in replays]
        state = [replay[1] for replay in replays]
        next_state = [replay[2] for replay in replays]
        reward = [replay[3] for replay in replays]
        done_bool = [replay[4] for replay in replays]
        return action,state,next_state,reward,done_bool
