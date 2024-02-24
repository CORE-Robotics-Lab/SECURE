import gymnasium as gym
import panda_gym
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
from custom_task import MyTask
from panda_gym.envs.robots.panda import Panda
import numpy as np


class MyRobotTaskEnv(RobotTaskEnv):
    """My robot-task environment."""

    def __init__(self, render=False):
        sim = PyBullet(render=render)
        task = MyTask(sim)
        robot = Panda(sim, base_position=np.array([-0.6, 0.0, 0.0]))
        super().__init__(robot, task)






env = MyRobotTaskEnv(render=True)

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # random action
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()  # wait a bit to give a realistic temporal rendering

    if terminated or truncated:
        observation, info = env.reset()
        env.render() # wait a bit to give a realistic temporal rendering