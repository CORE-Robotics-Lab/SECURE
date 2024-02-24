# import gymnasium as gym
import gym
import pickle
import panda_gym
import time
from manual_control import control_robo_keyboard

def main():
    filename = './init_dang_states.pickle'
    env = gym.make("PandaSafePush-v3", render=True)
    observation = env.reset()
    dang_states_ls = []
    for i in range(1000):
        action = control_robo_keyboard(observation)
        observation_crr = observation.copy()
        observation, reward, terminated, info = env.step(action)
        is_save_state = input(">> Save the states or not? [y/n]\n")
        if is_save_state == 'y':
            dang_states_ls.append(observation_crr)
        if len(dang_states_ls) % 10 == 0 and len(dang_states_ls) > 0:
            is_save = input(">> End and save or not? [y/n]\n")
            if is_save == 'y':
                with open(filename, 'wb') as handle:
                    pickle.dump(dang_states_ls, handle, protocol=pickle.HIGHEST_PROTOCOL)
                break

    env.close()



if __name__ == '__main__':
    main()