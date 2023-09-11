#!/usr/bin/env python3
import numpy as np

from envs.carEnv import carEnv
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from garage.envs import GymEnv
from garage.tf.policies import GaussianMLPPolicy
from global_utils.utils import *
import argparse
from garage.experiment import deterministic
from models.comb_core import *
# from models.train_cbf_panda_final import calc_deriv_batch
import models.config as config_cbf
import tensorflow_probability as tfp
import time
import gym
import panda_gym
import math
from models.architectures import relu_net
from models.train_cbf_nn_demolition_derby import racecar_nn





def parse_args():
    parser = argparse.ArgumentParser()
    seed = 10

    parser.add_argument('--demo_path', type=str,
                        default='src/demonstrations/demos_demolition_derby.pkl')
    parser.add_argument('--policy_path', type=str,
                        default='data/demolition_derby/airl_model/share')
    parser.add_argument('--airl_path', type=str,
                        default='data/demolition_derby/airl_model/airl')
    parser.add_argument('--cbf_pth', type=str, default="data/demolition_derby/cbf_model/cbf")
    parser.add_argument('--seed', type=int, required=False, default=seed)
    parser.add_argument('--alpha', type=float, required=False, default=10)
    parser.add_argument('--safe_ratio1', type=float, required=False, default=0.1)
    parser.add_argument('--safe_ratio2', type=float, required=False, default=0.3)
    parser.add_argument('--method', type=int, required=False, default=1)
    args = parser.parse_args()
    return args




def circle(samples=1000):
    points = []
    for i in range(samples):
        angle = ((math.pi * 2) / samples) * i
        points.append([math.cos(angle), math.sin(angle)])
    return points

def dynamic_model(s, a, bs):
    ob_batch = np.stack([s for _ in range(bs)], axis=0)  # [bs, 18]
    ob_batch_pos = ob_batch.copy()
    contact_mask = abs(ob_batch_pos[:, 0] - ob_batch_pos[:, 6]) < 0.15
    ob_batch_pos[contact_mask, 6:9] = ob_batch_pos[contact_mask, 6:9] + 0.033 * a[contact_mask, :]
    ob_batch_pos[:, :3] = ob_batch_pos[:, :3] + 0.033 * a
    ob_batch_pos[:, 3:6] = a / 5.0
    s_next = ob_batch_pos.copy()
    s = ob_batch.copy()
    return s, s_next


def sort_safe_actions(q_graph, s_ph, s_next_ph, a_ph, is_safe_graph, is_safe, actions, s, s_next, safe_ratio, method, sess):
    safe_idxes = np.where(is_safe)[0]
    safe_actions = actions[safe_idxes, :]
    action_m1 = np.mean(actions[safe_idxes, :], axis=0)
    comb_comp = (action_m1)
    return comb_comp[method-1], safe_actions.shape[0] == 0, comb_comp


# Global params
EVAL_TRAJ_NUM = 100
VIDEO_TRAJ_NUM = 20
INCREASE_VAR_LINE = 50
INCREASE_INTERVAL = 50
INCREASE_RATIO = 0.05
MAX_SAFE_WAIT_TIME = 100
BATCH_SIZE = 64  # ori -> panda: 128

SAMPLE_NUM=1000
CIRCLE_POINTS = circle(samples=SAMPLE_NUM)
RESAMPLE_NUM_NO_ADAPTIVE = 10

# Get params
args = parse_args()
log_path = args.policy_path
demo_pth = args.demo_path
# Set seeds
seed = args.seed
deterministic.set_seed(seed)

# Create env for graph construction
env = GymEnv(carEnv(demo=demo_pth), max_episode_length=1000)

# Config for tf.Session  &&  set GPU=0
os.environ["CUDA_VISIBLE_DEVICES"] = ""
config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True, log_device_placement=False)
with tf.Session(config=config) as sess:
    # policy graph construction
    policy = GaussianMLPPolicy(name=f'action',
                               env_spec=env.spec,
                               hidden_sizes=(32, 32))

    # Restore policy
    save_dictionary = {}
    for idx, var in enumerate(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                          scope=f'action')):
        save_dictionary[f'action_{idx}'] = var
    saver = tf.train.Saver(save_dictionary)
    saver.restore(sess, f"{log_path}/model")

    # Evaluation

    # Create test env
    env_test = carEnv(demo=demo_pth, is_test=True)
    env_test.render('no_vis')
    ob = env_test.reset()

    # Preparation
    imgs, coll_ls, succ_ls = [], [], []
    done, is_safe = False, False
    succ_cnt, traj_cnt, coll_cnt, cnt = 0, 0, 0, 0


    # Construct graph
    s_around = tf.placeholder(tf.float32, [SAMPLE_NUM, 2 * (config_cbf.TOP_K + 1)], name='states_around')  # [1000, 26]
    h_states_around = racecar_nn(s_around)
    is_safe_action, ph, deriv = calc_deriv_batch(args.alpha, config_cbf.DIST_MIN_THRES, BATCH_SIZE)  # use 10 for alpha (todo)
    s, s_next, a = ph

    # Restore CBF params
    cbf_path = args.cbf_pth
    save_dictionary_cbf = {}
    for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'cbf')):
        save_dictionary_cbf[f'cbf_{idx}'] = var
    print(">> Length of save_dictionary_cbf: ", len(save_dictionary_cbf))
    saver_cbf = tf.train.Saver(save_dictionary_cbf)
    if os.path.exists(cbf_path):
        print(">> Path exists!")
        saver_cbf.restore(sess, f"{cbf_path}/model")

    # tf MultivariateNormalDiag graph
    loc, scale_diag = tf.placeholder(tf.float32, [2,], name='loc'), tf.placeholder(tf.float32, [2,], name='scale_diag')
    dist = tfp.distributions.MultivariateNormalDiag(
        loc=loc,
        scale_diag=scale_diag)
    samples = dist.sample([BATCH_SIZE])

    # load q function model
    # process s & s_next to make them a batch size (special process for racecar domain)
    s_batch = tf.tile(tf.reshape(s, [1, -1]), [BATCH_SIZE, 1])
    s_next_batch = tf.tile(tf.reshape(s_next, [1, -1]), [BATCH_SIZE, 1])

    s_agent_batch = s_batch[:, -4:]
    dsdt = tf.concat([s_agent_batch[:, 2:], tf.reshape(a, [BATCH_SIZE, -1])], axis=1)
    s_next_agent_batch = s_agent_batch + dsdt * config_cbf.TIME_STEP
    s_next_batch = tf.concat([s_next_batch[:, :-4], s_next_agent_batch], axis=1)

    gamma = 0.99
    with tf.variable_scope('reward'):
        reward = relu_net(s_batch, dout=1)
    with tf.variable_scope('vfn'):
        fitted_value_fn_n = relu_net(s_next_batch, dout=1)
    q_graph = reward + gamma * fitted_value_fn_n

    # restore reward and vfn params
    save_dictionary_q = {}
    for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'reward')):
        save_dictionary_q[f'my_skill_0_{idx}'] = var

    for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'vfn')):
        save_dictionary_q[f'my_skill_0_{idx + 6}'] = var

    # restore reward params
    if os.path.exists(args.airl_path):
        print("Found Path!")
        saver = tf.train.Saver(save_dictionary_q)
        saver.restore(sess, f"{args.airl_path}/model")


    # time lists
    get_is_safe_tm_ls = []
    resample_times = []
    eval_traj_time, timepoint = [], time.time()
    get_safe_action_avg_ls = []
    traj_datas = []
    traj_data = {"observations": [], "actions": []}

    # Start eval
    fall_angle_ls = []
    is_fall = False
    fall_angle = 0
    a_cnt, a_pos_cnt, a_no_forward_cnt = 0, 0, 0
    a_comp_ls = []
    while traj_cnt < EVAL_TRAJ_NUM:
        # for one traj
        if not done:
            traj_data["observations"].append(ob)
            tp = time.time()
            cnt = 0
            action, m_v = policy.get_action(ob)
            mean, log_std = m_v['mean'], m_v['log_std']
            std = np.exp(log_std)
            mean_init, std_init = mean.copy(), std.copy()
            # get initial batch actions
            actions = []

            # batch actions
            for i in range(BATCH_SIZE):
                actions.append(policy.get_action(ob)[0])
            action = np.stack(actions)
            action_init = action.copy()  # [bs, 3]
            # resample actions
            while True:
                cnt += 1
                if cnt > 100:
                    msn = 1

                ob_next = env_test.get_next_state(action[0, :].reshape([-1]))

                get_is_safe_tm_s = time.time()

                # here input s & s_next are not batched cuz we only need the obstacles info in the s_next
                # the batched agent in the s_next will be calculated in the calc_deriv_batch function
                is_safe, deriv_ = sess.run([is_safe_action, deriv], feed_dict={
                    s: ob.reshape(config_cbf.TOP_K + 1, 4),
                    s_next: ob_next.reshape(config_cbf.TOP_K + 1, 4),
                    a: action.reshape([BATCH_SIZE, 1, -1])})

                get_is_safe_tm_e = time.time()
                get_is_safe_tm_ls.append(get_is_safe_tm_e - get_is_safe_tm_s)

                if traj_cnt <= 1:
                    adsads=1
                    if cnt > 20:
                        rf = 1

                if cnt > MAX_SAFE_WAIT_TIME:
                    action = action[0, :]
                    break


                # resample actions if unsafe. If safe, then break.
                # old: once find safe action, then break
                # new: need to find 10% * batch_size safe actions, then break (to increase robustness)
                # if np.any(is_safe):
                if np.sum(is_safe) >= int(args.safe_ratio1 * BATCH_SIZE):
                    action, no_forward_action, comb_comp = sort_safe_actions(q_graph, s, s_next, a, is_safe_action,
                                               is_safe, action.reshape([BATCH_SIZE, 1, -1]),
                                                 ob.reshape(config_cbf.TOP_K + 1, 4),
                                                 ob_next.reshape(config_cbf.TOP_K + 1, 4),
                                                 args.safe_ratio2, args.method, sess)

                    a_cnt += 1
                    a_comp_ls.append(comb_comp)
                    if action[0] > 0:
                        a_pos_cnt += 1
                    if no_forward_action:
                        a_no_forward_cnt += 1

                    # old version to get action
                    break
                elif cnt < RESAMPLE_NUM_NO_ADAPTIVE:
                    action = sess.run([samples], feed_dict={loc: mean, scale_diag: std_init})[0].reshape(BATCH_SIZE, -1)
                else:
                    # Detect which direction to increase std
                    R = 0.01
                    agent_pos = ob[-4:-2]
                    around_points = np.array(CIRCLE_POINTS) * R + np.tile(agent_pos, (len(CIRCLE_POINTS), 1))  # [SAMPLE_NUM, 2]
                    around_states = np.concatenate([np.tile(ob.reshape(config_cbf.TOP_K + 1, 4)[:-1, :2].reshape([-1]), (len(CIRCLE_POINTS), 1)), around_points], axis=1)  # [SAMPLE_NUM, 6]
                    around_states_h = sess.run(h_states_around, feed_dict={s_around: around_states}).reshape((SAMPLE_NUM, 1))  # [SAMPLE_NUM, 1]
                    coef = np.max(np.concatenate([-1 * around_states_h, np.zeros_like(around_states_h)], axis=1), axis=1, keepdims=True)  # [SAMPLE_NUM, 1]

                    around_points_no_transition = np.array(CIRCLE_POINTS) * R
                    std_delta = np.abs(np.sum((coef[:, None] * around_points_no_transition[..., None]).reshape(around_points_no_transition.shape[0], -1), axis=0))  # [2])
                    std_delta /= np.max(std_delta)
                    if np.all(coef == 0):
                        mean, std = mean, std_init + np.array([1, 1]) * 0.1 * cnt
                    else:
                        mean, std = mean, std_init + std_delta * 0.1 * cnt

                    num = 10
                    if std[0] > num and std[1] > num:
                        std = np.array([num, num])
                    action = sess.run([samples], feed_dict={loc: mean, scale_diag: std})[0].reshape(BATCH_SIZE, 1, 2)

            resample_times.append(cnt)
            ob, rew, done, info = env_test.step(action.reshape([-1]))
            is_safe = False
            get_safe_action_avg_ls.append(time.time() - tp)
            if traj_cnt < VIDEO_TRAJ_NUM:
                imgs.append(env_test.render('rgb_array'))
        else:
            new_timepoint = time.time()
            eval_traj_time.append(new_timepoint - timepoint)
            timepoint = new_timepoint
            print(">> Eval traj num: ", traj_cnt)
            traj_cnt += 1
            coll_ls.append(info['collision_num'])
            coll_cnt = coll_cnt + info['collision_num']
            succ_cnt = succ_cnt + info['success']
            print(">> is success: ", info['success'])
            print(">> collision num: ", info['collision_num'])
            ob = env_test.reset()
            done = False

    # Log info
    print(">> Success traj num: ", succ_cnt, ", Collision traj num: ", coll_cnt, " out of ", EVAL_TRAJ_NUM,
          " trajs.")
    print(coll_ls)
    print(str(np.mean(coll_ls)) + ', ' + str(np.std(coll_ls)))
    print(succ_ls)
    print("avg calc is_safe time:", np.mean(get_is_safe_tm_ls))
    print("avg resample times:", np.mean(resample_times))
    print("avg traj time:", np.mean(eval_traj_time), len(eval_traj_time), eval_traj_time)
    print("get_safe_action_avg_ls: ", np.mean(get_safe_action_avg_ls), len(get_safe_action_avg_ls))

    # Write log
    with open(log_path + "/eval_results_secure.txt", 'w', encoding='utf-8') as f:
        f.write(">> Success traj num: " + str(succ_cnt) + ", Collision traj num: " + str(coll_cnt) + " out of " + str(
            EVAL_TRAJ_NUM) + " trajs.\n")
        f.write(str(np.mean(coll_ls)) + ', ' + str(np.std(coll_ls)) + '\n')
        f.write(str(coll_ls))
