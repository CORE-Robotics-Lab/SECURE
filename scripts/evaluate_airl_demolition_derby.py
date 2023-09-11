#!/usr/bin/env python3
from envs.carEnv import carEnv
from garage.envs import GymEnv
from garage.tf.policies import GaussianMLPPolicy
from global_utils.utils import *
import argparse
from garage.experiment import deterministic
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', type=str, default='src/demonstrations/demos_demolition_derby.pkl')
    parser.add_argument('--policy_path', type=str, default='data/demolition_derby/airl_model/share')
    parser.add_argument('--seed', type=int, required=False, default=10)
    args = parser.parse_args()
    return args

args = parse_args()
log_path = args.policy_path
demo_pth = args.demo_path
EVAL_TRAJ_NUM = 100

# Set seeds
seed = args.seed
print("seed: ", seed)
deterministic.set_seed(seed)

irl_models = []
policies = []
algos = []
trainers = []

env = GymEnv(carEnv(demo=demo_pth), max_episode_length=1000)

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
    env_test = carEnv(demo=demo_pth, is_test=True)  # YY
    env_test.render('no_vis')
    imgs = []
    done = False
    ob = env_test.reset()
    succ_cnt, traj_cnt, coll_cnt, cnt = 0, 0, 0, 0
    video_traj_num = 10
    coll_ls, succ_ls = [], []
    last_timestep_state_ls = []
    eval_traj_time, timepoint = [], time.time()
    action_time_ls = []

    while traj_cnt < EVAL_TRAJ_NUM:
        if not done:
            get_action_s = time.time()
            ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
            get_action_e = time.time()
            action_time_ls.append(get_action_e - get_action_s)
        else:
            new_timepoint = time.time()
            eval_traj_time.append(new_timepoint - timepoint)
            timepoint = new_timepoint
            print(">> Eval traj num: ", traj_cnt)
            traj_cnt += 1
            coll_ls.append(info['collision_num'])
            coll_cnt = coll_cnt + info['collision_num']
            succ_cnt = succ_cnt + info['success']
            last_timestep_state_ls += env_test.unsafe_states
            ob = env_test.reset()
            done = False

    print("avg traj time:", np.mean(eval_traj_time), eval_traj_time)
    print("avg get each action time: ", np.mean(action_time_ls), len(action_time_ls))
    print(">> Success traj num: ", succ_cnt, ", Collision traj num: ", coll_cnt, " out of ", EVAL_TRAJ_NUM, " trajs.")
    print(coll_ls)
    print(str(np.mean(coll_ls)) + ', ' + str(np.std(coll_ls)))
    print(succ_ls)
    with open(log_path + "/eval_results_just_airl.txt", 'w', encoding='utf-8') as f:
        f.write(">> Success traj num: " + str(succ_cnt) + ", Collision traj num: " + str(coll_cnt) + " out of " + str(EVAL_TRAJ_NUM) + " trajs.\n")
        f.write(str(np.mean(coll_ls)) + ', ' + str(np.std(coll_ls)) + '\n')
        f.write(str(coll_ls))
