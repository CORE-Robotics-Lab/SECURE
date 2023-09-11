# from envs.carEnv import carEnv
from gym.envs import register

register(
     id='CarEnv-v0',
     entry_point='envs.carEnv:carEnv',
     # max_episode_steps=1000,
)