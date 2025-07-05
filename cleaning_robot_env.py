import gym
from gym import spaces
import numpy as np
from robot_setup import generate_room, spawn_robot
from symmetry import build_room_permutations
from utils import room_to_channels

class CleaningRobotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, width=8):
        super().__init__()
        self.width = width
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 3, width, width), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.room = None
        self.robot_pos = None
        self.robot_orientation = None

    def reset(self):
        self.room = generate_room(width=self.width, n_dirt=5)
        self.room = spawn_robot(self.room)
        self._update_robot_state()
        return self._get_observation()

    def step(self, action):
        move_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dx, dy = move_map[action]
        x, y = self.robot_pos
        nx, ny = x + dx, y + dy

        reward = 0
        done = False

        # Remove robot from curr pos
        self.room[x, y] = 1

        # Check if next pos valid 
        if 0 <= nx < self.width and 0 <= ny < self.width and self.room[nx, ny] in [1, -1]:
            if self.room[nx, ny] == -1:
                reward = 1
            self.robot_pos = (nx, ny)
        else:
            reward = -0.1
            nx, ny = x, y
            self.robot_pos = (nx, ny)

        self.room[nx, ny] = 10 + self.robot_orientation

        if not np.any(self.room == -1):
            done = True

        self._update_robot_state()
        return self._get_observation(), reward, done, {}

    def _update_robot_state(self):
        pos = np.argwhere(self.room >= 10)[0]
        self.robot_pos = tuple(pos)
        self.robot_orientation = self.room[self.robot_pos] - 10

    def _get_observation(self):
        perms = build_room_permutations(self.room)
        obs = np.stack([room_to_channels(p) for p in perms])
        return obs

    def render(self, mode="human"):
        print(self.room)