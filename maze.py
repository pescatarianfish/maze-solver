import pygame
import gym
from gym.spaces import Discrete
import numpy as np
import matplotlib.pyplot as plt


pygame.init()
class MazeSetup(gym.Env):

    def __init__(self):
        super(MazeSetup, self).__init__()
        #maze layout configuration 2= start 'S' 3=sub goal 'G' 4=end goal 'E'

        self.maze = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 3, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 4, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        self.rows = self.maze.shape[0]
        self.col = self.maze.shape[1]

        self.start_pos = np.where(self.maze == 2)
        self.current = np.array([self.start_pos[0][0], self.start_pos[1][0]])
        self.sub_pos = np.where(self.maze == 3)
        self.end_pos = np.where(self.maze == 4)

        self.sum_R = 0
        self.actions = 0

        self.agent = pygame.image.load('img/icons8-robot-2-64.png')
        self.agent = pygame.transform.scale(self.agent, (50, 50))

        self.achieved = False
        self.sub_gained = False

        self._action_to_direction = {
            0: np.array([0, 1]),# Right
            1: np.array([-1, 0]), # Up
            2: np.array([0, -1]),# Left
            3: np.array([1, 0]) # Down
        }

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(self.rows), gym.spaces.Discrete(self.col)))
        # size of gui window
        self.size= pygame.display.set_mode((self.col*50, self.rows*50))

    def reset(self):
        """resets the environment to default state"""
        self.current = np.array([self.start_pos[0][0], self.start_pos[1][0]])
        self.sum_R = 0
        self.actions = 0
        self.achieved = False
        self.sub_gained = False
        return self.current

    def step(self, action):
        """updating the agent actions and rewards """
        finish = False
        self.actions += 1
        pos = self.current + self._action_to_direction[action]
        y,x = pos
        if 0 <= y < self.rows and 0 <= x < self.col:
            if self.maze[y,x]!= 1:
                self.current = pos
        reward = -1



        if np.array_equal(self.current, [self.sub_pos[0][0], self.sub_pos[1][0]]):
            if not self.sub_gained:
                reward = 15
                self.sub_gained = True
                self.achieved = True
        else:
            reward = -1


        if np.array_equal(self.current, [self.end_pos[0][0], self.end_pos[1][0]]):
            if self.achieved:
                reward = 60

                finish = True
            else: # Penalty for trying to reach the end goal directly
                reward = -15

        self.sum_R += reward
        return self.current, reward, finish, {}

    def render(self, mode='human', path=None):
        """creates the environment"""
        #self.size.fill((255, 255, 255))

        for r in range(self.rows):
            for c in range(self.col):
                if self.maze[r,c] == 0: #open path
                    hue = (0, 0, 0)
                if self.maze[r,c] == 1: #wall
                    hue = 'indianred4'
                if self.maze[r,c] == 2: #start
                    hue = (0, 255, 0)
                if self.maze[r,c] == 3:#subgoal
                    hue = (0, 0, 255)
                if self.maze[r,c] == 4:#end
                    hue = (255, 0, 0)

                cell = pygame.Rect(c * 50, r * 50, 50, 50)
                pygame.draw.rect(self.size, hue, cell)


        if path:
            for pos in path:
                y, x = pos
                pygame.draw.rect(self.size, (255, 255, 0), (x * 50, y * 50, 50, 50), 5)

        self.size.blit(self.agent, (self.current[1] * 50, self.current[0] * 50))


        font = pygame.font.Font(None, 30)
        r_text = font.render(f"Total Reward: {self.sum_R}", True, (255, 255, 255))
        a_text = font.render(f"Total Actions: {self.actions}", True, (255, 255, 255))

        self.size.blit(r_text, (10, 10))
        self.size.blit(a_text, (10, 35))
        pygame.display.update()



