import numpy as np

class RandomModel:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def choose_action(self, state):
        return np.random.choice(self.num_actions)