# import gymnasium as gym
import gym
import pickle
import panda_gym
import time
from manual_control import control_robo_keyboard

import random
import numpy as np
seed = 101
random.seed(seed)
np.random.seed(seed)

def main():
    # env = gym.make('PandaReach-v3', render=True)
    # env = gym.make("PandaSafePushJoints-v3", render=True)]
    filename = './demonstrations_varied_height3.pickle'
    # filename = './yy_try/dangerous_cases.pickle'

    env = gym.make("PandaSafePush-v3", render=True)

    observation = env.reset()
    # time.sleep(10000)

    demonstrations = []
    demonstration = {"observations": [], "actions": []}
    for _ in range(1000):
        # action = env.action_space.sample()  # random action
        action = control_robo_keyboard(observation)
        demonstration["observations"].append(observation)
        demonstration["actions"].append(action)
        observation, reward, terminated, info = env.step(action)
        is_restart_collection = input(">> Restart or not[y/n]: ")
        if is_restart_collection == 'y':
            demonstration = {"observations": [], "actions": []}
            observation = env.reset()
        # print(">> outside terminated: ", terminated)


        # # collect unsafe cases
        # save = input("start to save [y/n]: ")
        # flag = False
        # if save == "y":
        #     flag = True

        if terminated:
        # if True:
        # if flag:
            print("==========================================================================================")
            while True:
                is_adopt = input(">> Adopt or not[y/n]: ")
                if is_adopt == "y":
                    demonstrations.append(demonstration)
                    break
                elif is_adopt == "n":
                    demonstration = {"observations": [], "actions": []}
                    break
                else:
                    print(">> Other key pressed!")
                    continue

            while True:
                continue_collect = input(">> Continue to collect demo or not [y/n]: ")
                if continue_collect == "y":
                    break
                elif continue_collect == "n":
                    with open(filename, 'wb') as handle:
                        pickle.dump(demonstrations, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    return
                else:
                    print(">> Other key pressed!")
                    continue
            observation = env.reset()
            # env.render()

    env.close()



if __name__ == '__main__':
    main()