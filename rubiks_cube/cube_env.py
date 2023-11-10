#import pygame
import numpy as np
import time
import gym


#from gym_rubiks_cube.envs.objects3D import Cube
#from gym_rubiks_cube.envs.render import Scene
#from gym_rubiks_cube.envs.functions import Sphere

'''
ACTION MAP:
0: Ui
1: U
2: Mhi
3: Mh
4: Di
5: D
6: Bi
7: B
8: 
9:
'''




class TransformCubeObject:
    def __init__(self) -> None:
        # fmt: off
        self.transformation_permutations = [np.arange(54) for _ in range(18)]
        # all of these actions are seen from the perspective of looking at the red side of the cube, with the white side facing upwards

        # action = 0: top-layer counter-clockwise rotation
        self.transformation_permutations[0][
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38]
        ] = [2, 5, 8, 1, 4, 7, 0, 3, 6, 36, 37, 38, 9, 10, 11, 18, 19, 20, 27, 28, 29]
        # action = 1: top-layer clockwise rotation
        self.transformation_permutations[1][
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38]
        ] = [6, 3, 0, 7, 4, 1, 8, 5, 2, 18, 19, 20, 27, 28, 29, 36, 37, 38, 9, 10, 11]
        # action = 2: middle-layer horizontal counter-clockwise rotation
        self.transformation_permutations[2][
            [12, 13, 14, 21, 22, 23, 30, 31, 32, 39, 40, 41]
        ] = [39, 40, 41, 12, 13, 14, 21, 22, 23, 30, 31, 32]
        # action = 3: middle-layer horizontal clockwise rotation
        self.transformation_permutations[3][
            [12, 13, 14, 21, 22, 23, 30, 31, 32, 39, 40, 41]
        ] = [21, 22, 23, 30, 31, 32, 39, 40, 41, 12, 13, 14]
        # action = 4: bottom-layer counter-clockwise rotation
        self.transformation_permutations[4][
            [45, 46, 47, 48, 49, 50, 51, 52, 53, 15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44]
        ] = [47, 50, 53, 46, 49, 52, 45, 48, 51, 42, 43, 44, 15, 16, 17, 24, 25, 26, 33, 34, 35]
        # action = 5: bottom-layer clockwise rotation
        self.transformation_permutations[5][
            [45, 46, 47, 48, 49, 50, 51, 52, 53, 15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44]
        ] = [51, 48, 45, 52, 49, 46, 53, 50, 47, 24, 25, 26, 33, 34, 35, 42, 43, 44, 15, 16, 17]
        # action = 6: back-layer counter-clockwise rotation
        self.transformation_permutations[6][
            [27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 20, 23, 26, 36, 39, 42, 45, 46, 47]
        ] = [33, 30, 27, 34, 31, 28, 35, 32, 29, 20, 23, 26, 47, 46, 45, 2, 1, 0, 36, 39, 42]
        # action = 7: back-layer clockwise rotation
        self.transformation_permutations[7][
            [27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 20, 23, 26, 36, 39, 42, 45, 46, 47]
        ] = [29, 32, 35, 28, 31, 34, 27, 30, 33, 42, 39, 36, 0, 1, 2, 45, 46, 47, 26, 23, 20]
        # action = 8: middle-layer counter-clockwise rotation
        self.transformation_permutations[8][
            [3, 4, 5, 19, 22, 25, 37, 40, 43, 48, 49, 50]
        ] = [19, 22, 25, 50, 49, 48, 5, 4, 3, 37, 40, 43]
        # action = 9: middle-layer clockwise rotation
        self.transformation_permutations[9][
                [3, 4, 5, 19, 22, 25, 37, 40, 43, 48, 49, 50]
        ] = [43, 40, 37, 3, 4, 5, 48, 49, 50, 25, 22, 19]
        # action = 10: front-layer counter-clockwise rotation
        self.transformation_permutations[10][
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 6, 7, 8, 18, 21, 24, 38, 41, 44, 51, 52, 53]
        ] = [11, 14, 17, 10, 13, 16, 9, 12, 15, 18, 21, 24, 53, 52, 51, 8, 7, 6, 38, 41, 44]
        # action = 11: front-layer clockwise rotation
        self.transformation_permutations[11][
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 6, 7, 8, 18, 21, 24, 38, 41, 44, 51, 52, 53]
        ] = [15, 12, 9, 16, 13, 10, 17, 14, 11, 44, 41, 38, 6, 7, 8, 51, 52, 53, 24, 21, 18]
        # action = 12: left-layer downwards rotation
        self.transformation_permutations[12][
            [36, 37, 38, 39, 40, 41, 42, 43, 44, 0, 3, 6, 9, 12, 15, 29, 32, 35, 45, 48, 51]
        ] = [42, 39, 36, 43, 40, 37, 44, 41, 38, 35, 32, 29, 0, 3, 6, 45, 48, 51, 15, 12, 9]
        # action = 13: left-layer upwards rotation
        self.transformation_permutations[13][
            [36, 37, 38, 39, 40, 41, 42, 43, 44, 0, 3, 6, 9, 12, 15, 29, 32, 35, 45, 48, 51]
        ] = [38, 41, 44, 37, 40, 43, 36, 39, 42, 9, 12, 15, 51, 48, 45, 6, 3, 0, 29, 32, 35]
        # action = 14: middle-layer vertical downwards rotation
        self.transformation_permutations[14][
            [1, 4, 7, 10, 13, 16, 28, 31, 34, 46, 49, 52]
        ] = [34, 31, 28, 1, 4, 7, 46, 49, 52, 16, 13, 10]
        # action = 15: middle-layer vertical upwards rotation
        self.transformation_permutations[15][
            [1, 4, 7, 10, 13, 16, 28, 31, 34, 46, 49, 52]
        ] = [10, 13, 16, 52, 49, 46, 7, 4, 1, 28, 31, 34]
        # action = 16: right-layer downwards rotation
        self.transformation_permutations[16][
            [18, 19, 20, 21, 22, 23, 24, 25, 26, 2, 5, 8, 11, 14, 17, 27, 30, 33, 47, 50, 53]
        ] = [20, 23, 26, 19, 22, 25, 18, 21, 24, 33, 30, 27, 2, 5, 8, 47, 50, 53, 17, 14, 11]
        # action = 17: right-layer upwards rotation
        self.transformation_permutations[17][
            [18, 19, 20, 21, 22, 23, 24, 25, 26, 2, 5, 8, 11, 14, 17, 27, 30, 33, 47, 50, 53]
        ] = [24, 21, 18, 25, 22, 19, 26, 23, 20, 11, 14, 17, 53, 50, 47, 8, 5, 2, 27, 30, 33]

        # fmt: on

    def __call__(self, current_state: np.ndarray, action: int) -> np.ndarray:
        return current_state[self.transformation_permutations[action]]

    def isSolved(self, current_state: np.ndarray) -> bool:
        temp = current_state.reshape((6, 9))
        return (temp.max(axis=1) - temp.min(axis=1) == 0).all()


class RubiksCubeEnv(gym.Env):
    def __init__(self) -> None:
        
        # internal variables
        self.color_vector = None
        self.transform = TransformCubeObject()
        self.structure = None
        self._done = None
        self.goal_state = np.array([[j for _ in range(9)] for j in range(6)]).reshape(54)

        # variables for rendering
        self.__setup_render = False
        self._scene = None
        self._sphere = None
        self._dis = None
        self._font = None
        self._look_point = None
        self._screen_width = 600
        self._screen_height = 600
        self._delta_theta, self._delta_phi = None, None
        self._time_last_frame = None
        self._rotation_step = 5

        # public variables
        self.max_steps = None
        self.steps_since_reset = None
        # TODO: fix this, so it's automatic
        self.cap_fps = 10
        self.scramble_params = None  # number of random steps to do when scrambling

        self.action_space = gym.spaces.Discrete(18)
        self.observation_space = gym.spaces.Box(
            low=0, high=5, shape=(54,), dtype=np.int64
        )

         
    def isSolved(self) -> bool:
        return self.transform.isSolved(self.color_vector)

    def scramble(self) -> None:
        for _ in range(self.scramble_params if self.scramble_params != None else 10):
            self.color_vector = self.transform(self.color_vector, np.random.randint(18))

        # check that cube is not in a solved state
        if self.scramble_params != 0 and self.isSolved():
            self.scramble()
            
    def init_state(self,state) -> np.ndarray:
        #Initialize state at some random state
        self.color_vector = state
        
        return self.color_vector
    
    def get_state(self) -> np.ndarray:
        return self.color_vector
        

    def reset(self) -> np.ndarray:
        # define a vector representing the color of each side
        # number from 0-5 are mapped to the colors in the following order white, red, blue, orange, green and yellow
        self.color_vector = np.array(
            [[j for _ in range(9)] for j in range(6)]
        ).flatten()
        self.transform = TransformCubeObject()
        self._done = False
        self.steps_since_reset = 0

        self.scramble()

        return self.color_vector
    
    def set_scramble(self,no_moves) -> None:
        
        self.scramble_params = no_moves
        
    def rev_action(self, action) -> int:
        
        if action%2 == 0:
            return action+1
        else:
            return action-1

    def step(self, action: int) -> np.ndarray:
        #if self._done:
            #gym.logger.warn(
            #    "You are calling 'step()' even though this "
            #    "environment has already returned done = True. You "
            #    "should always call 'reset()' once you receive 'done = "
            #    "True' -- any further steps are undefined behavior."
            #)

        self.color_vector = self.transform(self.color_vector, action)
        if self.__setup_render:
            axis = action // 6
            index = action % 6 // 2
            # rotate either counter-clockwise or clockwise (if action % 2 == 0 -> counter-clockwise, else clockwise)
            flip_axis = action % 2
            step_rot = self.rotation_step if action % 2 == 0 else -self.rotation_step

            # show animation of doing a rotation
            for _ in range(0, 90, abs(step_rot)):
                if axis == 0:
                    self._scene.rotateObjects(
                        self.structure[index].flatten(), 2, np.deg2rad(step_rot)
                    )
                elif axis == 1:
                    self._scene.rotateObjects(
                        self.structure[:, index].flatten(), 1, np.deg2rad(-step_rot)
                    )
                else:
                    self._scene.rotateObjects(
                        self.structure[:, :, index].flatten(), 0, np.deg2rad(step_rot)
                    )
                self.render()

            # update render structure
            if axis == 0:
                self.structure[index] = np.flip(self.structure[index].T, axis=flip_axis)
            elif axis == 1:
                self.structure[:, index] = np.flip(
                    self.structure[:, index].T, axis=flip_axis
                )
            else:
                self.structure[:, :, index] = np.flip(
                    self.structure[:, :, index].T, axis=1 - flip_axis
                )

        self.steps_since_reset += 1

        if self.isSolved():
            self._done = True
            reward = 1
        elif self.max_steps != None and self.steps_since_reset >= self.max_steps:
            self._done = True
            reward = -0
        else:
            reward = -0

        return self.color_vector, reward, self._done, {}

    def close(self) -> None:
        if self.__setup_render:
            self.__setup_render = False
            self._scene = None
            self._sphere = None
            self._dis = None
            self._font = None
            self._look_point = None
            self._delta_theta, self._delta_phi = None, None


    @property
    def rotation_step(self) -> int:
        return self._rotation_step

    @rotation_step.setter
    def rotation_step(self, value):
        assert (
            type(value) == int and 90 % value == 0 and value > 0
        ), "rotation step must be an integer, bigger than 0 and divide 90"
        self._rotation_step = value