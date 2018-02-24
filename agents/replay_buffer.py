import random
from collections import namedtuple

Experience = namedtuple("Experience",
        field_names=["state", "action", "reward", "state_next", "done"])

class ReplayBuffer:

    def __init__(self, size=1024):
        self.size = size
        self.memory = []
        self.ptr = 0

    def add_experience(self, state, action, reward, state_next, done):
        exp = Experience(state, action, reward, state_next, done)
        if len(self.memory) < self.size:
            self.memory.append(exp)
        else:
            if self.ptr == self.size:
                self.ptr = 0
            self.memory[self.ptr] = exp
        self.ptr += 1

    def sample(self, batch_size=64):
        return random.sample(self.memory, k=batch_size)

    def len(self):
        return len(self.memory)
