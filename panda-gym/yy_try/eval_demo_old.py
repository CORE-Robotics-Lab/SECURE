import pickle
import panda_gym
import time
from manual_control import control_robo_keyboard
import os
import gym

import random
import numpy as np
seed = 1
random.seed(seed)
np.random.seed(seed)

filename = 'demonstrations.pickle'
filename = 'dangerous_cases.pickle'
filename = './demonstrations_varied_height.pickle'


def save_video(ims, filename, fps=15.0):
    import cv2
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()


with open(filename, 'rb') as handle:
    demos = pickle.load(handle)


print(len(demos))

env = gym.make("PandaSafePush-v3", render=True)
observation = env.reset()

imgs = []
for demo in demos:
    cnt = 0
    for i in range(len(demo["observations"])):
        if cnt > 70:
            continue
        img = env.render(mode="rgb_array")
        imgs.append(img[:, :, :3])
        action = demo["actions"][i]
        observation, reward, terminated, info = env.step(action)
        time.sleep(0.3)
        cnt += 1
    observation = env.reset()

save_video(imgs, "./demonstrations_varied_height.avi")