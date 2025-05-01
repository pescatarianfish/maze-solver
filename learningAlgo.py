
import numpy as np

class QLearningAgent:
    """ Q learning agent.
    book: Practical reinforcement learning with python"""

    def __init__(self, rows, cols, num_actions, alpha=None, gamma=None, epsilon=None, epsilon_decay=None, epsilon_min=None):
        """init q learning agent"""
        self.q_table = np.zeros((rows, cols, num_actions)) #blank q table
        self.epsilon = epsilon      #epsilon
        self.epsilon_decay = epsilon_decay #epsilon decay rate
        self.epsilon_min = epsilon_min #min epsilon
        self.num_actions = num_actions #total action
        self.alpha = alpha  # alpha
        self.gamma = gamma  #discount factor


    def choose_action(self, state):
        """choose an action based on policy"""
        y, x = state
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(self.q_table[y, x, :])
        return action

    def update_q_value(self, state, action, reward, next_state, done):
        """update q value"""
        y, x = state
        next_y, next_x = next_state
        best_next_action = np.argmax(self.q_table[next_y, next_x, :])
        #temporal difference target and error
        td_target = reward + self.gamma * self.q_table[next_y, next_x, best_next_action] * (not done)
        td_error = td_target - self.q_table[y, x, action]
        self.q_table[y, x, action] += self.alpha * td_error



    def decay_epsilon(self):
        """decay epsilon"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

