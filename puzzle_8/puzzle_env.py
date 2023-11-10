"""
8 Puzzle problem (or sliding tiles) involves solving a randomly generated board that looks like
|1|3|5|
|8|2|6|
|7|0|4|
by moving around the blank (0) tile with adjacent tiles using actions 0 (swap up), 1 (swap right), 2 (swap left), and
3 (swap down) until the following position is reached
|1|2|3|
|4|5|6|
|7|8|0|

The reward structure is +100 if the goal state is reached, else -1 for every step taken

"""

# Source. - https://github.com/JKCooper2/gym-envs/blob/master/EightPuzzle/eight_puzzle.py

import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from six import StringIO
import sys

logger = logging.getLogger(__name__)


class FourPuzzle(gym.Env):
    metadata = {
        'render.modes': ['ansi', 'human']
    }

    def __init__(self):
        self.size = 2
        #self.size = 3
        self.state = self._make_puzzle()
        #self.goal_state = np.array([[1,2,3],[4,5,6],[7,8,0]])
        self.goal_state = np.array([[1,2],[3,0]])
        self.action_space = spaces.Discrete(4)
        #self.observation_space = spaces.Box(0, 9, (9, ))
        self.observation_space = spaces.Box(0, 4, (4, ))                           
                                   

        self.last_action = None
        self.last_reward = None

        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _make_puzzle(self):
        #tiles = range(self.size ** 2)  # 0 is blank
        tiles = np.arange(self.size ** 2)
        np.random.shuffle(tiles)

        while not self._is_solvable(tiles):
            np.random.shuffle(tiles)

        return np.array(tiles).reshape((self.size, self.size))

    @staticmethod
    def _is_solvable(puzzle):
        # Based on http://math.stackexchange.com/questions/293527/how-to-check-if-a-8-puzzle-is-solvable
        inversions = 0

        for i in range(len(puzzle)):
            for j in range(i + 1, len(puzzle)):
                if puzzle[i] > puzzle[j] and puzzle[i] != 0 and puzzle[j] != 0:
                    inversions += 1

        return inversions % 2 == 0
    
    def init_state(self,state):
        #self.state = state.reshape(3,3)
        self.state = state.reshape(2,2)

    def rev_action(self,action):
        if action == 0:
            return 2
        if action == 1:
            return 3
        if action == 2:
            return 0
        if action == 3:
            return 1

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Update puzzle state
        blank = np.where(self.state == 0)
        b_x = blank[0][0]
        b_y = blank[1][0]

        # Move Up
        if action == 0 and b_x > 0:
            self.state[b_x][b_y] = self.state[b_x - 1][b_y]
            self.state[b_x - 1][b_y] = 0

        # Move Right
        elif action == 1 and b_y < 1:   # Change to 2 is 3x3 puzzle
            self.state[b_x][b_y] = self.state[b_x][b_y + 1]
            self.state[b_x][b_y + 1] = 0

        # Move Down
        elif action == 2 and b_x < 1:   # Change to 2 is 3x3 puzzle
            self.state[b_x][b_y] = self.state[b_x + 1][b_y]
            self.state[b_x + 1][b_y] = 0

        # Move Left
        elif action == 3 and b_y > 0:
            self.state[b_x][b_y] = self.state[b_x][b_y - 1]
            self.state[b_x][b_y - 1] = 0

            
        # Determine binary reward (if solved reward = 100, else -1)
        reward = 0
        
        done = (self.state == self.goal_state).all()

        
        # Determine reward (negative Manhattan distance to goal state)
        # distance = 0

        #for i, row in enumerate(self.state):
        #    for j, tile in enumerate(row):
        #        correct_row = np.floor(tile / self.size) - (1 if tile % self.size == 0 else 0)
        #        correct_col = (tile % self.size - 1) % self.size

        #        if tile == 0:
        #            correct_row = self.size - 1
        #            correct_col = self.size - 1

        #        distance += abs(i - correct_row) + abs(j - correct_col)

        #done = distance == 0
        #reward = -distance

        self.last_action = action
        self.last_reward = reward

        if done:
            reward = 1
            #if self.steps_beyond_done is None:
                #self.steps_beyond_done = 0
            #else:
                #logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        return self.state.flatten(), reward, done, {}

    def reset(self):
        self.state = self._make_puzzle()
        self.steps_beyond_done = None
        return self.state.flatten()
    
    def get_state(self):
        return self.state.flatten()

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write("\n".join('|'.join(map(str, row)).replace('0', ' ') for row in self.state) + "\n")
        if self.last_action is not None:
            outfile.write("{0} {1}\n".format(self.last_reward, ["U", "R", "D", "L"][self.last_action]))
            outfile.write("\n")
        else:
            outfile.write("\n")

        return outfile
    
    
 #Starting 8 puzzle environment

class EightPuzzle(gym.Env):
    metadata = {
        'render.modes': ['ansi', 'human']
    }

    def __init__(self):
        self.size = 3
        self.state = self._make_puzzle()
        self.goal_state = np.array([[1,2,3],[4,5,6],[7,8,0]])

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 9, (9, ))

        self.last_action = None
        self.last_reward = None

        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _make_puzzle(self):
        #tiles = range(self.size ** 2)  # 0 is blank
        tiles = np.arange(self.size ** 2)
        np.random.shuffle(tiles)

        while not self._is_solvable(tiles):
            np.random.shuffle(tiles)

        return np.array(tiles).reshape((self.size, self.size))

    @staticmethod
    def _is_solvable(puzzle):
        # Based on http://math.stackexchange.com/questions/293527/how-to-check-if-a-8-puzzle-is-solvable
        inversions = 0

        for i in range(len(puzzle)):
            for j in range(i + 1, len(puzzle)):
                if puzzle[i] > puzzle[j] and puzzle[i] != 0 and puzzle[j] != 0:
                    inversions += 1

        return inversions % 2 == 0
    
    def init_state(self,state):
        self.state = state.reshape(3,3)

    def rev_action(self,action):
        if action == 0:
            return 2
        if action == 1:
            return 3
        if action == 2:
            return 0
        if action == 3:
            return 1

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Update puzzle state
        blank = np.where(self.state == 0)
        b_x = blank[0][0]
        b_y = blank[1][0]

        # Move Up
        if action == 0 and b_x > 0:
            self.state[b_x][b_y] = self.state[b_x - 1][b_y]
            self.state[b_x - 1][b_y] = 0

        # Move Right
        elif action == 1 and b_y < 2:
            self.state[b_x][b_y] = self.state[b_x][b_y + 1]
            self.state[b_x][b_y + 1] = 0

        # Move Down
        elif action == 2 and b_x < 2:
            self.state[b_x][b_y] = self.state[b_x + 1][b_y]
            self.state[b_x + 1][b_y] = 0

        # Move Left
        elif action == 3 and b_y > 0:
            self.state[b_x][b_y] = self.state[b_x][b_y - 1]
            self.state[b_x][b_y - 1] = 0

            
        # Determine binary reward (if solved reward = 100, else -1)
        reward = 0
        
        done = (self.state == self.goal_state).all()

        
        # Determine reward (negative Manhattan distance to goal state)
        # distance = 0

        #for i, row in enumerate(self.state):
        #    for j, tile in enumerate(row):
        #        correct_row = np.floor(tile / self.size) - (1 if tile % self.size == 0 else 0)
        #        correct_col = (tile % self.size - 1) % self.size

        #        if tile == 0:
        #            correct_row = self.size - 1
        #            correct_col = self.size - 1

        #        distance += abs(i - correct_row) + abs(j - correct_col)

        #done = distance == 0
        #reward = -distance

        self.last_action = action
        self.last_reward = reward

        if done:
            reward = 1
            #if self.steps_beyond_done is None:
            #    self.steps_beyond_done = 0
            #else:
            #    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")

        return self.state.flatten(), reward, done, {}

    def reset(self):
        self.state = self._make_puzzle()
        self.steps_beyond_done = None
        return self.state.flatten()
    
    def get_state(self):
        return self.state.flatten()

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write("\n".join('|'.join(map(str, row)).replace('0', ' ') for row in self.state) + "\n")
        if self.last_action is not None:
            outfile.write("{0} {1}\n".format(self.last_reward, ["U", "R", "D", "L"][self.last_action]))
            outfile.write("\n")
        else:
            outfile.write("\n")

        return outfile