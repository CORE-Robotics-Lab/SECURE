import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random
import envs.config as config
import pickle
import os
from gym import Env, spaces
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from models.architectures import relu_net
import tensorflow as tf
from garage.envs import GymEnv
from garage.experiment import deterministic

class carEnv(Env):
    def __init__(self, demo='src/demonstrations/safe_demo_better.pkl', seed=10, is_hundred=False, is_test=False):
        super(carEnv, self).__init__()

        # random.seed(seed)

        # read demonstrations
        with open(demo, 'rb') as f:
            self.demonstrations = pickle.load(f)
            # if is_test:
            #     self.demonstrations = self.demonstrations[500:]  # yy: use last 500 for testing
            # else:
            #     self.demonstrations = self.demonstrations[:500]  # yy: use first 500 for training

        obs_num = self.demonstrations[0]['observations'][0].shape[0]

        # observation_space: (min(obs_num, TOP_K d_obs) + 1 agent) * [posx, posy, vx, vy]
        observation_shape = ((min(obs_num, config.TOP_K + 1)) * 4,)
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=observation_shape)
        # action_space: (ax, ay)
        self.action_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(2,))

        # initialize the states of agent and obs
        self.agent_state = None
        self.obs_state = None

        # traj's id
        self.__traj_id = None
        self.traj = None
        self.timestep = 0
        self.goal_state = None

        # create a canvas
        plt.ion()
        plt.close()
        self.fig = plt.figure(figsize=(9, 9))
        self.agent_size = None

        # vis
        self.is_vis = False

        self.max_episode_length = 1000

        # special setting for 100 per
        self.is_hundred = is_hundred

        self.collision_flag = False





    def reset(self):
        # self.seed(deterministic.get_seed())
        self.unsafe_states = []  # yy: store states before collision, return it when done
        self.success = 0
        self.collision_num = 0
        self.timestep = 0
        self.step_num = 0
        id = random.choice(range(len(self.demonstrations)))
        self.__traj_id = id
        self.traj = self.demonstrations[self.__traj_id]
        start_state, self.goal_state = self.traj['observations'][0], self.traj['observations'][-1][-1, :]
        self.agent_state, self.obs_state = start_state[-1, :], start_state[:-1, :]
        # self.obs_state = self.remove_dist_obs(self.agent_state, self.obs_state)

        vis_range = max(1, np.amax(np.abs(start_state[:, :2])))
        self.agent_size = 100 / vis_range ** 2
        if self.is_vis:
            self.draw_fig()

        # concatenate obv
        self.obs_state_removed = self.remove_dist_obs(self.agent_state, self.obs_state)
        obv = np.concatenate([self.obs_state_removed, self.agent_state[None, :]], axis=0).flatten()



        if self.is_hundred:
            full_obv = np.concatenate([self.obs_state, self.agent_state[None, :]], axis=0).flatten()
            return obv, full_obv
        else:
            return obv


    def step(self, action):
        self.timestep += 1
        self.step_num += 1

        if self.step_num > 54:
            opp = 1

        if self.timestep >= len(self.traj['observations']):
            self.timestep = len(self.traj['observations']) - 1

        # step of the agent
        dsdt = np.concatenate([self.agent_state[2:], action])  # YY: dsdt = [vx, vy, ax, ay]
        self.agent_state = self.agent_state + dsdt * config.TIME_STEP

        # step of the obs
        self.obs_state = self.traj['observations'][self.timestep][:-1, :]
        self.obs_state_removed = self.remove_dist_obs(self.agent_state, self.obs_state)

        # concatenate obv
        obv = np.concatenate([self.obs_state_removed, self.agent_state[None, :]], axis=0).flatten()

        # # next observation
        # obs_next = self.obs_state = self.traj['observations'][self.timestep + 1][:-1, :]

        done = False

        reward = - np.linalg.norm(self.agent_state[:2] - self.goal_state[:2])
        if self.step_num >= 100:
            done = True

        if np.linalg.norm(self.agent_state[:2] - self.goal_state[:2]) < config.DIST_SUCC:
            self.success = 1
            self.collision_flag = False

        elif not np.all(np.linalg.norm(self.obs_state[:, :2] - self.agent_state[:2], axis=1) > config.DIST_MIN_CHECK):
            if not self.collision_flag:
                self.collision_num += 1
                self.collision_flag = True
        else:
            self.collision_flag = False

        if self.is_vis:
            self.draw_fig()

        return obv, reward, done, {'success': self.success,
                                   'collision_num': self.collision_num}


    def get_next_state(self, action):
        next_timestep = self.timestep + 1
        if self.timestep >= len(self.traj['observations']) - 1:
            next_timestep = len(self.traj['observations']) - 1

        # step of the agent
        # print(self.agent_state[2:].shape, action.shape)
        dsdt = np.concatenate([self.agent_state[2:], action])  # YY: dsdt = [vx, vy, ax, ay]
        agent_state = self.agent_state + dsdt * config.TIME_STEP

        # step of the obs
        obs_state = self.traj['observations'][next_timestep][:-1, :]
        obs_state_removed = self.remove_dist_obs(agent_state, obs_state)

        # concatenate obv
        obv = np.concatenate([obs_state_removed, agent_state[None, :]], axis=0).flatten()

        return obv



    def render(self, mode="rgb_array"):
        # TODO
        if mode == "human":
            self.is_vis = True
        if mode == 'no_vis':
            self.is_vis = False
        if mode == "rgb_array":
            img = self.draw_fig()
            self.is_vis = False
            return img


    def close(self):
        # TODO
        pass
        # plt.clf()


    def draw_fig(self):
        plt.clf()

        # show the dynamic obstacles
        plt.scatter(self.obs_state[:, 0], self.obs_state[:, 1],
                    color='darkorange',
                    s=self.agent_size, label='Dynamic Obstacle', alpha=0.6)
        # show the agentsafe_demo_better
        plt.scatter(self.agent_state[0], self.agent_state[1],
                    color='green',
                    s=self.agent_size, label='Agent', alpha=0.6)

        plt.scatter(self.goal_state[0], self.goal_state[1],
                    color='deepskyblue',
                    s=self.agent_size, label='Target', alpha=0.6)


        plt.xlim(-0.5, 2.0)
        plt.ylim(-0.5, 2.0)

        self.fig.canvas.draw()

        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        # image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if not self.is_vis:
            return image
        else:
            plt.pause(0.0001)


    def get_traj_id(self):
        return self.__traj_id


    # YY: agent_state with shape (4), obs_states with shape (obs_num, 4)
    def remove_dist_obs(self, agent_state, obs_state):
        topk_mask = np.argsort(np.sum(np.square((obs_state - agent_state)[:, :2]), axis=1))[:config.TOP_K]
        return obs_state[topk_mask, :]


    # return shape [agent_num, state_dim]
    def get_start_goal_states(self):
        self.traj = self.demonstrations[self.__traj_id]
        return self.traj['observations'][0], self.traj['observations'][-1][:, :2]

    def set_traj(self, demo):
        self.traj = demo







def save_video(ims, filename, fps=30.0):
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





if __name__ == "__main__":
    # load reward
    log_path = f"data/yy/22_07_2022_17_00_38"  # r(s, a)
    # log_path = f"data/yy/25_07_2022_18_57_41"  # r(s)


    # log_path = f'data/obs16/03_08_2022_16_26_14'  # r(s) -> 1000 epoch
    log_path = 'data/obs16/03_08_2022_22_59_25'  # r(s, a) -> 1000 epoch

    # log_path = f'data/obs16/04_08_2022_17_23_23'  # r(s) -> 400 epoch
    log_path = 'data/obs16/05_08_2022_14_49_34'  # r(s, a) -> 400 epoch


    demo_pth = 'src/demonstrations/safe_demo_16obs_stop.pkl'
    dim = 52

    save_dictionary = {}
    env = GymEnv(carEnv(demo=demo_pth), max_episode_length=50)
    with tf.Session() as sess:
        ph_obv = tf.placeholder(tf.float32, shape=(dim), name='ph_obv')
        a = tf.placeholder(tf.float32, shape=(2,), name='ph_a')
        rew_input = tf.reshape(ph_obv, [1, dim])  # r(s)
        rew_input = tf.concat([tf.reshape(ph_obv, [1, dim]), tf.reshape(a, [1, 2])], axis=1)  # r(s, a)
        with tf.variable_scope('skill_0/discrim/reward'):
            # loss_reward = tf.reduce_sum(relu_net(rew_input, dout=1, **{}))
            loss_reward = relu_net(rew_input, dout=1, **{})

        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'skill_0/discrim/reward')):
            # print(var.name)
            # print(var)
            save_dictionary[f'reward_0_{idx}'] = var

        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{log_path}/model")




        # Get agent's traj's actions
        with open(demo_pth, 'rb') as f:
            demonstrations = pickle.load(f)


        NUM_DEMO = 100
        reward_all_traj = []
        for i in range(NUM_DEMO):
            print('Iter ', i)
            env = carEnv(demo=demo_pth)
            obs = env.reset()
            agent_actions = demonstrations[env.get_traj_id()]['actions']
            agent_actions = demonstrations[i]['actions']

            tm = 0
            # env.render()
            reward_one_traj = []
            while True:
                # Take a random action
                # action = env.action_space.sample()
                # print(tm)
                if tm >= len(agent_actions):
                    tm = -1
                action = agent_actions[tm][-1, :]
                # print('a: ', action)
                # print('v: ', obs)

                # action = np.array([0.01, 0.01])
                obs, reward, done, info = env.step(action)
                # print(obs, action)

                # Render the game
                # env.render()

                # loss_reward_ = sess.run(loss_reward, {ph_obv: obs})
                loss_reward_ = sess.run(loss_reward, {ph_obv: obs, a: action})
                # print(loss_reward_[0][0])
                reward_one_traj.append(loss_reward_[0][0])

                if info['success'] == 1:
                    break

                if done == True:
                    break

                tm += 1

            # # show the reward map
            # start_obs, end_obs = env.get_start_goal_states()
            # start_obs, end_obs = start_obs[-1, :2], end_obs[-1, :]
            # for i in range(start_obs[0], end_obs[0], 0.1):
            #     for j in range(start_obs[1], end_obs[1], 0.1):

            reward_all_traj.append(reward_one_traj)
            env.close()

        with open('data/debug_data/reward_s_a_16obs_stop_400epoch.pkl', 'wb') as f:
            pickle.dump(reward_all_traj, f)










    #
    # # Get agent's traj's actions
    # with open('src/demonstrations/safe_demo_32obs.pkl', 'rb') as f:
    #     demonstrations = pickle.load(f)
    # imgs = []
    # for i in range(10, 20):
    #     env = carEnv()
    #     obs = env.reset()
    #
    #
    #     agent_actions = demonstrations[env.get_traj_id()]['actions']
    #     # for i in range(100):
    #     #     agent_actions = demonstrations[i]['actions']
    #     #     print(len(agent_actions))
    #
    #     agent_actions = demonstrations[i]['actions']
    #     # print(len(agent_actions))
    #     #
    #     # print(agent_actions[0].shape)
    #
    #     tm = 0
    #     env.render('human')
    #
    #     while True:
    #         imgs.append(env.render('rgb_array'))
    #         # Take a random action
    #         # action = env.action_space.sample()
    #         # print(tm)
    #         if tm >= len(agent_actions):
    #             tm = -1
    #         action = agent_actions[tm][-1, :]
    #         # print('a: ', action)
    #         # print('v: ', obs)
    #
    #         # action = np.array([0.01, 0.01])
    #         obs, reward, done, info = env.step(action)
    #         # print(obs, action)
    #
    #         # Render the game
    #         # env.render()
    #
    #
    #         if done == True:
    #             break
    #
    #         tm += 1
    #
    #
    #     env.close()
    # # save_video(imgs, f"data/tmp_videos/demostration_stop.avi")