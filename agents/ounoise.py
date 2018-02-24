import numpy as np

class OUNoise:

    def __init__(self, size, mu=1.2, theta=1.0, dt = 0.001, sigma=0.3):
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        '''Reset the internal state'''
        self.state = self.mu

    def sample(self):
        '''Update internal state and return it as a noise sample'''
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * \
                np.random.randn(self.size)
        self.state = x + dx
        return self.state
