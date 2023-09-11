#!/usr/bin/env python3
import numpy as np


# from envs.carEnv_garage import carEnv
import os
from datetime import datetime
import gym
import seaborn as sns
import matplotlib.pyplot as plt
# import tensorflow as tf
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler, MultiprocessingSampler
from airl.irl_trpo import TRPO
from models.airl_state import AIRL

from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import Trainer
from global_utils.utils import *
from garage.experiment import Snapshotter
import pickle
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import argparse
from garage.experiment import deterministic
from models.architectures import relu_net
import time
import panda_gym

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def parse_args():
    seed=10
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_path', type=str, default='data/panda_arm_push/airl_model/share')
    parser.add_argument('--seed', type=int, required=False, default=seed)
    args = parser.parse_args()
    return args

args = parse_args()
log_path = args.policy_path
EVAL_TRAJ_NUM = 100

# Set seeds
seed = args.seed
print("seed: ", seed)
deterministic.set_seed(seed)

irl_models = []
policies = []
algos = []
trainers = []

env = GymEnv(gym.make("PandaSafePush-v3", render=False), max_episode_length=50, is_panda=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    save_dictionary = {}
    policy = GaussianMLPPolicy(name=f'action',
                               env_spec=env.spec,
                               hidden_sizes=(32, 32))
    for idx, var in enumerate(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                          scope=f'action')):
        save_dictionary[f'action_{idx}'] = var

    policies.append(policy)
    saver = tf.train.Saver(save_dictionary)
    saver.restore(sess, f"{log_path}/model")

    # Evaluation
    env_test = gym.make("PandaSafePush-v3", render=True)
    imgs = []
    done = False
    ob = env_test.reset()
    succ_cnt, fall_cnt, traj_cnt, cnt = 0, 0, 0, 0
    video_traj_num = 20
    succ_ls = []
    eval_traj_time, timepoint = [], time.time()
    action_time_ls = []
    traj_obvs = []
    is_fall = False
    fall_angle = 0
    fall_angle_ls = []
    traj_datas = []
    traj_data = {"observations": [], "actions": []}

    while traj_cnt < EVAL_TRAJ_NUM:
        if not done:
            traj_data["observations"].append(ob)
            get_action_s = time.time()
            action = policy.get_action(ob)[0]
            ob, rew, done, _, info = env_test.step(action)
            traj_data["actions"].append(action)

            if info['is_fall']:
                is_fall = True
            fall_angle = max(info['fall_angle'], fall_angle)
            if traj_cnt == 0:
                traj_obvs.append(ob)
            get_action_e = time.time()
            action_time_ls.append(get_action_e - get_action_s)
        else:
            new_timepoint = time.time()
            eval_traj_time.append(new_timepoint - timepoint)
            timepoint = new_timepoint

            traj_data["observations"] = np.array(traj_data["observations"])
            traj_data["actions"] = np.array(traj_data["actions"])
            traj_datas.append(traj_data)
            traj_data = {"observations": [], "actions": []}

            print(">> Eval traj num: ", traj_cnt)
            print(">> Is fall: ", is_fall)
            print(">> Is success: ", info['is_success'])
            print(">> Fall angle: ", fall_angle)
            fall_angle_ls.append(fall_angle)
            traj_cnt += 1
            if not is_fall:
                succ_cnt = succ_cnt + info['is_success']

            fall_cnt = fall_cnt + is_fall
            ob = env_test.reset()
            done = False
            is_fall = False
            fall_angle = 0

    suffix = "_video"
    print("avg traj time:", np.mean(eval_traj_time), eval_traj_time)
    print("avg get each action time: ", np.mean(action_time_ls), len(action_time_ls))

    print(">> Success traj num: ", succ_cnt, " out of ", EVAL_TRAJ_NUM,
          " trajs.")
    print(">> Fall num: ", fall_cnt)
    print(">> Fall angle: ", np.mean(fall_angle_ls), fall_angle_ls)
    print(succ_ls)
    with open(log_path + "/eval_results_just_airl.txt", 'w', encoding='utf-8') as f:
        f.write(">> Success traj num: " + str(succ_cnt) + ", fall num: " + str(fall_cnt) + " out of " + str(EVAL_TRAJ_NUM) + " trajs.\n")
        # f.write(str(fall_angle_ls))
        f.write("\n>> Fall angle avg: " + str(np.mean(fall_angle_ls)))
