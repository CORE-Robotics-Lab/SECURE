import sys

sys.dont_write_bytecode = True

import os
import h5py
import argparse
import numpy as np
import tensorflow as tf

import models.comb_core as comb_core
import models.config as config

# -------------------------------- new import --------------------------------
# from garage.tf.policies import GaussianMLPPolicy
from envs.carEnv import carEnv
from garage.envs import GymEnv
import pickle
from garage.experiment import deterministic
import random
from datetime import datetime
from models.architectures import relu_net
import envs.config as config_file
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(4)


def parse_args():
    parser = argparse.ArgumentParser()
    now = datetime.now()
    parser.add_argument('--goal_reaching_weight', type=float, required=False, default=0.1)
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--share_path', type=str, default=None)
    parser.add_argument('--cbf_path', type=str, default=None)
    parser.add_argument('--demo_pth', type=str, default='src/demonstrations/safe_demo_16obs_stop.pkl')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--is_hundred', type=int, default=0)
    parser.add_argument('--training_epoch', type=int, required=False, default=20000)
    args = parser.parse_args()
    return args


def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_list.append(np.mean(acc[acc[:, i] >= 0, i]))
    return acc_list

# # Generate unsafe states
# def generate_unsafe_states(S_s, A_s, num_ratio=0.1, num_unsafe_state_each_frame=3):
#     DIST_MIN_THRES = 0.2
#     topk = 3  # topk obs will be selected
#     # randomly select some frames to create unsafe states
#     mask = random.sample(range(len(S_s)), int(num_ratio * len(S_s)))
#     S_u_init, A_u_init = np.array(S_s)[mask], np.array(A_s)[mask]
#     S_u, A_u = [], []
#     for i, s_u in enumerate(S_u_init):
#         # randomly select one obstacle
#         # TODO: only choose obs that are close to the current agent
#         topk_mask = np.argsort(np.sum(np.square((s_u[:-1, :] - s_u[-1, :])[:, :2]), axis=1))[:topk]
#         to_sele_states = s_u[:-1, :][topk_mask, :]
#         for j in range(topk):
#             rand_idx = topk_mask[j]
#             # rand_idx = random.choice(range(s_u[:-1, :].shape[0]))
#             # create num_unsafe_state_each_frame unsafe stdemo_pthates around the selected obstacle
#             for _ in range(topk - j):  # select topk for the closest, topk - 1 for the 2nd closest, ...
#                 s_agent = s_u[:-1, :]
#                 s_agent = s_agent[rand_idx, :]
#                 # add bias based on the state of the selected obstacle's state, then we get the unsafe state of our agent
#                 axis_range = np.random.choice(np.linspace(DIST_MIN_THRES / 3, DIST_MIN_THRES, 10), 1)[0] / np.sqrt(2)
#                 x_direction, y_direction = 2 * random.random() - 1, 2 * random.random() - 1
#                 x_bias, y_bias = x_direction * axis_range, y_direction * axis_range
#                 v_range, a_range = 1, 2  # the range of velocity and acceleration
#                 s_agent = np.array(
#                     [s_agent[0] + x_bias, s_agent[1] + y_bias, -1 * x_direction * v_range, -1 * y_direction * v_range])
#                 a_agent = np.array([-1 * x_direction * a_range, -1 * y_direction * a_range])
#                 # combine the unsafe state of agent with other states
#                 s_agent = np.concatenate([s_u[:-1, :], s_agent.reshape(1, -1)], axis=0)
#
#                 a_agent = np.concatenate([A_u_init[i][:-1, :], a_agent.reshape(1, -1)], axis=0)
#                 S_u.append((s_agent, rand_idx))
#                 A_u.append(a_agent)
#     return S_u, A_u


def generate_unsafe_states(S_s, A_s, num_ratio=0.1, num_unsafe_state_each_frame=3):
    DIST_MIN_THRES = 0.06
    topk = 3  # topk obs will be selected
    # randomly select some frames to create unsafe states
    mask = random.sample(range(len(S_s)), int(num_ratio * len(S_s)))
    S_u_init, A_u_init = np.array(S_s)[mask], np.array(A_s)[mask]
    S_u, A_u = [], []
    for i, s_u in enumerate(S_u_init):
        # randomly select one obstacle
        # TODO: only choose obs that are close to the current agent
        topk_mask = np.argsort(np.sum(np.square((s_u[:-1, :] - s_u[-1, :])[:, :2]), axis=1))[:topk]
        to_sele_states = s_u[:-1, :][topk_mask, :]
        for j in range(topk):
            rand_idx = topk_mask[j]
            # create num_unsafe_state_each_frame unsafe states around the selected obstacle
            s_agent = s_u[:-1, :]
            s_agent = s_agent[rand_idx, :]
            # add bias based on the state of the selected obstacle's state, then we get the unsafe state of our agent
            # axis_range = np.random.choice(np.linspace(DIST_MIN_THRES / 3, DIST_MIN_THRES, 10), 1)[0] / np.sqrt(2)
            # axis_range = (((2 * random.random() - 1) / 100) * 5 + 0.15) / np.sqrt(2)
            # yy: old generation -> 16obs
            axis_range = np.random.choice([0.01, 0.03, 0.07, 0.1, 0.125, 0.15, 0.2], 1, p=[0.3, 0.225, 0.175, 0.1, 0.1, 0.05, 0.05])[0] / np.sqrt(2)

            axis_range = \
            np.random.choice([0.01, 0.03, 0.07], 1, p=[0.5, 0.3, 0.2])[
                0] / np.sqrt(2)

            x_direction, y_direction = 2 * random.random() - 1, 2 * random.random() - 1
            x_bias, y_bias = x_direction * axis_range, y_direction * axis_range
            v_range, a_range = 1, 2  # the range of velocity and acceleration
            # unsafe agent's state and action
            # s_agent = np.array([s_agent[0] + x_bias, s_agent[1] + y_bias])  # posx, posy
            s_agent = np.array([s_agent[0] + x_bias, s_agent[1] + y_bias, -1 * x_direction * v_range, -1 * y_direction * v_range])   # posx, posy, velx, vely
            a_agent = np.array([-1 * x_direction * v_range, -1 * y_direction * v_range])
            # combine the unsafe state of agent with other states
            s_agent = np.concatenate([s_u[:-1, :], s_agent.reshape(1, -1)], axis=0)
            a_agent = np.concatenate([A_u_init[i][:-1, :], a_agent.reshape(1, -1)], axis=0)
            S_u.append((s_agent, rand_idx))
            A_u.append(a_agent)
    return S_u, A_u


# Generate unsafe states in simple cases
def generate_simple_unsafe_states(num_obs, goal_state, unsafe_states_num):
    S_u, A_u = [], []
    side_lgth = goal_state[0]
    par_lgth = side_lgth / (num_obs + 1)
    obs_state = np.zeros([num_obs, 4])
    for i in range(num_obs):
        tmp = np.zeros([1, 4])
        tmp[0, 0] = (i + 1) * par_lgth
        tmp[0, 1] = (i + 1) * par_lgth
        obs_state[i, :] = tmp
    # randomly select one obstacle
    rand_idx = random.choice(range(obs_state.shape[0]))
    # create num_unsafe_state_each_frame unsafe stdemo_pthates around the selected obstacle
    num_unsafe_state_each_frame = unsafe_states_num
    for _ in range(num_unsafe_state_each_frame):
        s_agent = obs_state
        s_agent = s_agent[rand_idx, :]
        # add bias based on the state of the selected obstacle's state, then we get the unsafe state of our agent
        axis_range = config.DIST_MIN_THRES / np.sqrt(2)
        x_direction, y_direction = 2 * random.random() - 1, 2 * random.random() - 1
        x_bias, y_bias = x_direction * axis_range, y_direction * axis_range
        v_range, a_range = 1, 2  # DONE TODO: check the range of velocity and acceleration

        s_agent = np.array(
            [s_agent[0] + x_bias, s_agent[1] + y_bias, -1 * x_direction * v_range, -1 * y_direction * v_range])

        # s_agent = np.array(
        #     [s_agent[0] + x_bias, s_agent[1] + y_bias, -1 * (2 * random.random() - 1) * v_range, -1 * (2 * random.random() - 1) * v_range])

        a_agent = np.array([-1 * x_direction * a_range, -1 * y_direction * a_range])
        # combine the unsafe state of agent with other states
        s_agent = np.concatenate([obs_state, s_agent.reshape(1, -1)], axis=0)
        S_u.append((s_agent, rand_idx))
        A_u.append(a_agent)
    return S_u, A_u


def demo_remove_top_k(demos, topk):
    for i, demo in enumerate(demos):
        obvs = demo['observations']
        for j, obv in enumerate(obvs):
            topk_mask = np.argsort(np.sum(np.square((obv[:-1, :] - obv[-1, :])[:, :2]), axis=1))[:topk]
            demos[i]['observations'][j] = np.concatenate([obv[:-1, :][topk_mask, :], obv[-1, :][None, :]], axis=0)
    return demos


def get_action_graph(ob, policy):
    # samples = policy._f_dist(ob)

    # dist, mean, log_std = policy.build(ob, name='action_cbf' + str(istep)).outputs
    # samples = dist.sample(seed=deterministic.get_tf_seed_stream())
    # return tf.reshape(samples, [1, 2])

    return tf.reshape(policy.get_action(ob)[0], [1, 2])


def get_action(ob, policy, seed):
    # dist, mean, log_std = policy.build(ob, name='default_cbf').outputs
    dist, mean, log_std = policy.extend(ob).outputs
    # samples = dist.sample(seed=deterministic.get_tf_seed_stream())
    samples = dist.sample()
    # samples = dist.mean()
    return tf.reshape(samples, [1, 2]), mean


def get_action_complex(ob, policy):
    dist, mean, log_std = policy.extend(ob).outputs
    # samples = dist.sample(seed=deterministic.get_tf_seed_stream())
    samples = dist.sample()
    return tf.reshape(samples, [1, 2])


def get_action_batch(ob, policy, bs):
    return tf.compat.v1.map_fn(fn=lambda t: tf.reshape(policy.extend(t).outputs[0].sample(), [1, 2]), elems=ob)


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if not f]

    # print(str(i.name) for i in not_initialized_vars)  # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def ground_truth_reward(pos_next_agent, g):
    # a = s_next_agent[0, :2]
    # b = g[0, :2]
    # return -((a[0] - b[0])**2 + (a[1] - b[1])**2)

    return -tf.norm(pos_next_agent - g[-1:, :2])


def build_optimizer(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE)
    trainable_vars = tf.trainable_variables()

    # tensor to accumulate gradients over multiple steps
    accumulators = [
        tf.Variable(
            tf.zeros_like(tv.initialized_value()),
            trainable=False
        ) for tv in trainable_vars]

    # count how many steps we have accumulated
    accumulation_counter = tf.Variable(0.0, trainable=False)
    grad_pairs = optimizer.compute_gradients(loss, trainable_vars)
    # add the gradient to the accumulation tensor
    accumulate_ops = [
        accumulator.assign_add(
            grad
        ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]

    accumulate_ops.append(accumulation_counter.assign_add(1.0))
    # divide the accumulated gradient by the number of accumulation steps
    gradient_vars = [(accumulator / accumulation_counter, var) \
                     for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]
    # seperate the gradient of CBF and the controller
    gradient_vars_h = []
    gradient_vars_a = []
    for accumulate_grad, var in gradient_vars:
        print(var.name)
        if 'cbf' in var.name:
            gradient_vars_h.append((accumulate_grad, var))
        elif 'action' in var.name:
            gradient_vars_a.append((accumulate_grad, var))
        elif 'reward' in var.name:
            print(">> [INFO] Do not update reward params")
            continue
        else:
            print(">> [INFO] Param not updated in cbf module")
            continue

    train_step_h = optimizer.apply_gradients(gradient_vars_h)
    # train_step_a = optimizer.apply_gradients(gradient_vars_a)
    train_step_a = None
    # re-initialize the accmulation tensor and accumulation step to zero
    zero_ops = [
        accumulator.assign(
            tf.zeros_like(tv)
        ) for (accumulator, tv) in zip(accumulators, trainable_vars)]
    zero_ops.append(accumulation_counter.assign(0.0))

    return zero_ops, accumulate_ops, train_step_h, train_step_a


def build_optimizer_deriv(loss, lr=1e-4):
    # lr scheduler
    # starter_learning_rate = lr
    # learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
    #                                                      global_step,
    #                                                      100000, 0.96, staircase=True)


    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    trainable_vars = tf.trainable_variables()





    new_vars = []
    for var in trainable_vars:
        # if "log_std_network" in var.name:
        #     continue
        # if "cbf" in var.name:
        #     continue
        if "action" in var.name:
            new_vars.append(var)

    trainable_vars = new_vars





    # tensor to accumulate gradients over multiple steps
    accumulators = [
        tf.Variable(
            tf.zeros_like(tv.initialized_value()),
            trainable=False
        ) for tv in trainable_vars]

    # count how many steps we have accumulated
    accumulation_counter = tf.Variable(0.0, trainable=False)
    grad_pairs = optimizer.compute_gradients(loss, trainable_vars)
    # add the gradient to the accumulation tensor
    accumulate_ops = [
        accumulator.assign_add(
            grad
        ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]

    accumulate_ops.append(accumulation_counter.assign_add(1.0))
    # divide the accumulated gradient by the number of accumulation steps
    gradient_vars = [(accumulator / accumulation_counter, var) \
                     for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]
    # seperate the gradient of CBF and the controller
    gradient_vars_h = []
    gradient_vars_a = []
    # print('--------------------------------')
    for accumulate_grad, var in gradient_vars:
        # print(var.name)
        if 'cbf' in var.name:
            gradient_vars_h.append((accumulate_grad, var))
        elif 'action' in var.name:
            gradient_vars_a.append((accumulate_grad, var))
        elif 'reward' in var.name:
            # print(">> [INFO] Do not update reward params")
            continue
        else:
            # print(">> [INFO] Param not updated in cbf module")
            continue
    # print('--------------------------------')

    train_step_a = optimizer.apply_gradients(gradient_vars_a)
    # re-initialize the accmulation tensor and accumulation step to zero
    zero_ops = [
        accumulator.assign(
            tf.zeros_like(tv)
        ) for (accumulator, tv) in zip(accumulators, trainable_vars)]
    zero_ops.append(accumulation_counter.assign(0.0))

    return zero_ops, accumulate_ops, train_step_a


def build_optimizer_simple(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE)
    trainable_vars = tf.trainable_variables()
    # new_vars = []
    # for var in trainable_vars:
    #     # if "log_std_network" in var.name:
    #     #     continue
    #     if "cbf" in var.name:
    #         continue
    #     new_vars.append(var)
    #
    # trainable_vars = new_vars
    [print(var.name) for var in trainable_vars]

    # tensor to accumulate gradients over multiple steps
    accumulators = [
        tf.Variable(
            tf.zeros_like(tv.initialized_value()),
            trainable=False
        ) for tv in trainable_vars]

    # count how many steps we have accumulated
    accumulation_counter = tf.Variable(0.0, trainable=False)
    grad_pairs = optimizer.compute_gradients(loss, trainable_vars)
    grads = [g for g, var in grad_pairs]


    # for grad, var in grad_pairs:
    #     print(var.name)
    #
    # print('----------------')
    # add the gradient to the accumulation tensor
    accumulate_ops = [
        accumulator.assign_add(
            grad
        ) for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]

    accumulate_ops.append(accumulation_counter.assign_add(1.0))
    # divide the accumulated gradient by the number of accumulation steps
    gradient_vars = [(accumulator / accumulation_counter, var)
                     for (accumulator, (grad, var)) in zip(accumulators, grad_pairs)]
    # seperate the gradient of CBF and the controller
    gradient_vars_h = []
    gradient_vars_a = []
    for accumulate_grad, var in gradient_vars:
        print(var.name)
        if 'cbf' in var.name:
            gradient_vars_h.append((accumulate_grad, var))
        elif 'action' in var.name:
            gradient_vars_a.append((accumulate_grad, var))
        elif 'reward' in var.name:
            # print(">> [INFO] Do not update reward params")
            continue
        else:
            # print(">> [INFO] Param not updated in cbf module")
            continue
    print(gradient_vars_a)
    train_step_h = optimizer.apply_gradients(gradient_vars_h)
    # train_step_h = None
    train_step_a = optimizer.apply_gradients(gradient_vars_a)
    # re-initialize the accmulation tensor and accumulation step to zero
    zero_ops = [
        accumulator.assign(
            tf.zeros_like(tv)
        ) for (accumulator, tv) in zip(accumulators, trainable_vars)]
    zero_ops.append(accumulation_counter.assign(0.0))

    return zero_ops, accumulate_ops, train_step_h, train_step_a, grads


def build_training_graph_init(num_obs):
    """
    Description:
        Train CBF NN. Only minimizing loss_barrier.

    Args:
        num_obs: number of obstacles

    Returns:

    """
    # Placeholders
    s = tf.placeholder(tf.float32, [min(num_obs + 1, config.TOP_K + 1), 4], name='ph_state')  # state
    dang_mask_reshape = tf.placeholder(tf.bool, [min(num_obs + 1, config.TOP_K + 1), ], name='ph_dang_mask')  # dang mask
    safe_mask_reshape = tf.placeholder(tf.bool, [min(num_obs + 1, config.TOP_K + 1), ], name='ph_safe_mask')  # safe mask

    # x is difference between the state of each agent and other agents
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)  # yy: shape: [13, 13, 4]

    # Get h() from CBF NN (h shape: 1 * 13)
    h, mask = comb_core.network_cbf(x=x, r=config.DIST_MIN_THRES)

    # Get loss_safe and loss_dang and corresponding accuracy
    (loss_dang, loss_safe, acc_dang, acc_safe) = comb_core.loss_barrier(h=h,
                                                                        dang_mask_reshape=dang_mask_reshape,
                                                                        safe_mask_reshape=safe_mask_reshape)

    # Compute objective loss
    loss_list = [2 * loss_dang, loss_safe]
    acc_list = [acc_dang, acc_safe]
    weight_loss = [config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]  # Weight decay
    loss = 10 * tf.math.add_n(loss_list + weight_loss)

    return s, dang_mask_reshape, safe_mask_reshape, h, loss, loss_dang, loss_safe, acc_dang, acc_safe, loss_list, acc_list


def build_training_graph_deriv(num_obs, policy):
    """
    Description:
        Train policy to be safer. Freeze CBF NN and only minimizing loss_deriv.

    Args:
        num_obs: number of obstacles
        policy: the policy for the agent

    Returns:

    """
    # Placeholders
    s = tf.placeholder(tf.float32, [min(num_obs + 1, config.TOP_K + 1), 4], name='ph_state')  # state
    s_next = tf.placeholder(tf.float32, [min(num_obs + 1, config.TOP_K + 1), 4],
                            name='ph_state_next')  # next timestep state

    # x is difference between the state of each agent and other agents
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)  # yy: shape: [13, 13, 4]

    # Get h() from CBF NN (h shape: 1 * 13). yy: freeze CBF NN here!!
    # h, mask = tf.stop_gradient(comb_core.network_cbf(x=x, r=config.DIST_MIN_THRES))  # TODO: check how to freeze
    h, mask = comb_core.network_cbf(x=x, r=config.DIST_MIN_THRES)

    # Get action via the policy
    # a = get_action_graph(tf.reshape(s, [1, 4 * min(num_obs + 1, config.TOP_K + 1)]), policy, istep)
    # a = get_action_graph(ob, policy)
    a = get_action_complex(tf.reshape(s, [1, 4 * min(num_obs + 1, config.TOP_K + 1)]), policy)

    # Get loss and accuracy for derivative part
    loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv, num_dang, num_safe, unsafe_states = \
        comb_core.loss_derivatives(s, a, h, s_next, r=config.DIST_MIN_THRES, alpha=config.ALPHA_CBF)

    # Compute objective loss
    loss_list = [2 * loss_dang_deriv, loss_safe_deriv]
    acc_list = [acc_dang_deriv, acc_safe_deriv]
    # weight_loss = [config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]  # Weight decay todo: check
    # loss = 10 * tf.math.add_n(loss_list + weight_loss)
    loss = 10 * tf.math.add_n(loss_list)

    return s, s_next, h, loss, loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv, loss_list, acc_list, num_dang, num_safe, unsafe_states


def build_training_graph_deriv_min(num_obs, policy):
    """
    Description:
        Train policy to be safer. Freeze CBF NN and only minimizing loss_deriv.

    Args:
        num_obs: number of obstacles
        policy: the policy for the agent

    Returns:

    """
    # Placeholders
    s = tf.placeholder(tf.float32, [min(num_obs + 1, config.TOP_K + 1), 4], name='ph_state')  # state
    s_next = tf.placeholder(tf.float32, [min(num_obs + 1, config.TOP_K + 1), 4],
                            name='ph_state_next')  # next timestep state

    # x is difference between the state of each agent and other agents
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)  # yy: shape: [13, 13, 4]

    # Get h() from CBF NN (h shape: 1 * 13). yy: freeze CBF NN here!!
    h, mask = comb_core.network_cbf(x=x, r=config.DIST_MIN_THRES)

    # Get action via the policy
    a = get_action_complex(tf.reshape(s, [1, 4 * min(num_obs + 1, config.TOP_K + 1)]), policy)

    # Get loss and accuracy for derivative part
    loss_deriv, acc_deriv, h_min, h_next_min, deriv_reshape = \
        comb_core.loss_derivatives_min(s, a, h, s_next, r=config.DIST_MIN_THRES, alpha=config.ALPHA_CBF)

    # Compute objective loss
    loss_list = [loss_deriv]
    # weight_loss = [config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]  # Weight decay todo: check
    # loss = 10 * tf.math.add_n(loss_list + weight_loss)
    loss = 10 * tf.math.add_n(loss_list)

    return s, s_next, h, loss, acc_deriv, h_min, h_next_min, deriv_reshape



# yy: for pure cbf & cbf with ground truth reward function
def build_training_graph_cbf_ground_truth(num_obs, g, policy, seed):
    """
    Description:
        Train policy to be safer. Freeze CBF NN and only minimizing loss_deriv.

    Args:
        num_obs: number of obstacles
        g: the goal

    Returns:

    """
    # Placeholders
    s = tf.placeholder(tf.float32, [min(num_obs + 1, config.TOP_K + 1), 4], name='ph_state')  # state
    s_next = tf.placeholder(tf.float32, [min(num_obs + 1, config.TOP_K + 1), 4],
                            name='ph_state_next')  # next timestep state  # todo, think of how to create this!!!
    dang_mask_reshape = tf.placeholder(tf.bool, [min(num_obs + 1, config.TOP_K + 1), ], name='ph_dang_mask')  # dang mask
    safe_mask_reshape = tf.placeholder(tf.bool, [min(num_obs + 1, config.TOP_K + 1), ], name='ph_safe_mask')  # safe mask
    goal = tf.placeholder(tf.float32, [1, 4])


    # x is difference between the state of each agent and other agents
    x = tf.expand_dims(s, 1) - tf.expand_dims(s, 0)  # yy: shape: [13, 13, 4]

    # Get h() from CBF NN (h shape: 1 * 13).
    h, mask = comb_core.network_cbf(x=x, r=config.DIST_MIN_THRES)

    # Get action via the policy
    a, m = get_action(tf.reshape(s, [1, 4 * min(num_obs + 1, config.TOP_K + 1)]), policy, seed)

    # Get loss_safe and loss_dang and corresponding accuracy
    (loss_dang, loss_safe, acc_dang, acc_safe) = comb_core.loss_barrier(h=h,
                                                                        dang_mask_reshape=dang_mask_reshape,
                                                                        safe_mask_reshape=safe_mask_reshape)

    # Get loss and accuracy for derivative part
    loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv = \
        comb_core.loss_derivatives(s, a, h, s_next, r=config.DIST_MIN_THRES, alpha=config.ALPHA_CBF)


    # Calculate ground truth loss
    '''
    s, goal are tf.placeholder: 
    - s[-1, :] -> [pos_x, pos_y, v_x, v_y]
    - goal -> [pos_x_goal, pos_y_goal, v_x_goal, v_y_goal]
    - a -> [a_x, a_y]   |   a is action, computed by poilcy(s)
    '''
    dsdt = tf.concat([tf.reshape(s[-1, 2:], (1, 2)), tf.reshape(a, (1, 2))], axis=1)  # dsdt = [v_x, v_y, a_x, a_y]
    next_state = tf.reshape((s[-1, :] + dsdt * config.TIME_STEP), (1, 4))  # next_state = [pos_x, pos_y, v_x, v_y] + delta_time * [v_x, v_y, a_x, a_y]
    next_position = next_state[-1:, :2] + next_state[-1:, 2:] * config.TIME_STEP  # next_position = [pos_x_next, pos_y_next] + delta_time * [v_x_next, v_y_next]
    loss_distance = tf.compat.v1.norm(goal[-1:, :2] - next_position)  # distance between next position and goal position
    # loss_distance = tf.compat.v1.norm(a)


    # loss_action = -ground_truth_reward(agent_pos, goal)

    # grad if grad is not None else tf.zeros_like(var)


    # Compute objective loss
    loss_list = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv]
    acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]
    # weight_loss = [config.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()]  # Weight decay todo: check
    # loss = 10 * tf.math.add_n(loss_list + weight_loss)
    loss = 10 * tf.math.add_n(loss_list)
    # loss = loss_distance

    return s, s_next, dang_mask_reshape, safe_mask_reshape, h, loss, loss_list, acc_list, loss_dang, loss_safe, acc_dang, acc_safe, loss_distance, goal, a, m


def train_init_CBF_NN(demo_path,
                      log_path,
                      cbf_save_path,
                      num_obs,
                      is_load_unsafe_states=False,
                      unsafe_ratio=1,
                      unsafe_state_path='src/demonstrations/unsafe_states_16obs_close.pkl'):
    """
    Description:
        Train initial CBF NN

    Args:
        demo_path: path for demonstrations
        log_path: path for log files of tensorboard
        cbf_save_path: path for saving CBF NN
        num_obs: number of obstacles
        is_load_unsafe_states: load unsafe states or not
        unsafe_ratio: ratio of #unsafe states to #safe states
        unsafe_state_path: path for unsafe states

    Returns:
        None
    """

    # Tensorboard logger
    log_path = log_path
    writer = SummaryWriter(log_path)

    # Load demonstrations
    with open(demo_path, 'rb') as f:
        demonstrations = pickle.load(f)

    # Only retain topk nearest obs in demonstrations
    demonstrations = demo_remove_top_k(demonstrations, config.TOP_K)

    # Goal state
    goal_state = demonstrations[0]['observations'][-1][-1, :]

    # Get set of states and actions from demonstrations
    S_s = [s for traj in demonstrations for s in traj['observations']]
    A_s = [a for traj in demonstrations for a in traj['actions']]

    # Collect unsafe states. Load if already exist.
    if is_load_unsafe_states:
        with open(unsafe_state_path, 'rb') as f:
            S_u, A_u = pickle.load(f)
    else:
        # # 3 obs -> simple case
        # S_u, _ = generate_simple_unsafe_states(num_obs, goal_state, 20000)
        # with open(unsafe_state_path, 'wb') as f:
        #     pickle.dump((S_u, _), f)

        # 16 obs
        S_u, A_u = generate_unsafe_states(S_s, A_s, num_ratio=unsafe_ratio)
        with open(unsafe_state_path, 'wb') as f:
            pickle.dump([S_u, A_u], f)


    print(">> Ori total safe states num: ", len(S_s))
    print(">> Ori total unsafe states num: ", len(S_u))
    # Shuffle
    np.random.shuffle(S_s)
    np.random.shuffle(S_u)
    # # Only use parts of S_s and S_u
    S_s = S_s[:20000]
    S_u = S_u[:20000]

    # Add idx -1 to each s_s S_s
    S_s = [(s_s, -1) for s_s in S_s]
    # Divide S_s, S_u
    traj_length = 50
    S_s_eval = S_s[len(S_s) // 2:]
    S_u_eval = S_u[len(S_u) // 2:]
    S_s = S_s[:len(S_s) // 2]
    S_u = S_u[:len(S_u) // 2]
    TRAIN_STEPS = len(S_s) // traj_length
    unsafe_length = len(S_u) // TRAIN_STEPS
    print(">> Total training steps: ", TRAIN_STEPS)
    print(">> Training | Safe states num: ", len(S_s), ", Unsafe states num: ", len(S_u))
    print(">> Evaluation | Safe states num: ", len(S_s_eval), ", Unsafe states num: ", len(S_u_eval))

    # Start training
    print("---------- Start Training ----------")
    with tf.Session() as sess:
        # Construct training graph
        s, dang_mask_reshape, safe_mask_reshape, h, loss, loss_dang, loss_safe, acc_dang, acc_safe, loss_list, acc_list = build_training_graph_init(num_obs)
        zero_ops, accumulate_ops, train_step_h, train_step_a = build_optimizer(loss)
        accumulate_ops.append(loss_list)
        accumulate_ops.append(acc_list)
        # Initialize global variables
        sess.run(tf.global_variables_initializer())
        # Prepare for saving CBF NN
        cbf_path = cbf_save_path
        save_dictionary_cbf = {}
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'cbf')):
            save_dictionary_cbf[f'cbf_{idx}'] = var
        saver_cbf = tf.train.Saver(save_dictionary_cbf)

        # Preparation
        loss_lists_np = []
        acc_lists_np = []
        # Iterate training steps
        for istep in range(TRAIN_STEPS):
            print('>> Step: ', istep)
            # Each iteration's amount of safe and unsafe states
            S_s_iter = S_s[istep * traj_length: (istep + 1) * traj_length]
            S_u_iter = S_u[istep * unsafe_length: (istep + 1) * unsafe_length]
            # Re-initialize accumulators for each iteration
            sess.run(zero_ops)

            # Main training parts
            # Mix up S_s & S_u
            for i in range(len(S_s_iter) + len(S_u_iter)):
                if len(S_s_iter) == 0:
                    s_, idx = S_u_iter.pop(0)
                elif len(S_u_iter) == 0:
                    s_, idx = S_s_iter.pop(0)
                elif random.random() > .5:  # select one safe state
                    s_, idx = S_s_iter.pop(0)
                else:
                    s_, idx = S_u_iter.pop(0)

                # Create dan & safe mask
                k = min(num_obs + 1, config.TOP_K + 1)
                dang_mask_reshape_ = np.zeros([k]).astype(bool)
                safe_mask_reshape_ = np.ones([k]).astype(bool)
                safe_mask_reshape_[-1] = False
                if idx != -1:  # no dangerous h
                    dang_mask_reshape_[idx] = True
                    safe_mask_reshape_[idx] = False

                out, loss_, loss_dang_, loss_safe_, acc_dang_, acc_safe_ =\
                    sess.run([accumulate_ops, loss, loss_dang, loss_safe, acc_dang, acc_safe],
                             feed_dict={s: s_, dang_mask_reshape: dang_mask_reshape_, safe_mask_reshape: safe_mask_reshape_})

                # Original codebase way to add accuracy and loss
                loss_list_np, acc_list_np = out[-2], out[-1]
                loss_lists_np.append(loss_list_np)
                acc_lists_np.append(acc_list_np)
            print("Step: ", istep, ", loss: ", np.mean(loss_lists_np, axis=0))
            writer.add_scalar('train_loss_dang', np.mean(loss_lists_np, axis=0)[0], istep)
            writer.add_scalar('train_loss_safe', np.mean(loss_lists_np, axis=0)[1], istep)
            # Actual Optimization
            sess.run(train_step_h)

            # Evaluation
            EVAL_STEPS, EVAL_RATIO = 10, 0.5
            if np.mod(istep, EVAL_STEPS) == 0:
                s_u_eval, s_s_eval = \
                    random.sample(S_u_eval, int(len(S_u_eval) * EVAL_RATIO)), random.sample(S_s_eval, int(len(S_s_eval) * EVAL_RATIO))
                # s_u_eval, s_s_eval = [(s_u, 0) for s_u in s_u_eval], [(s_s, 1) for s_s in s_s_eval]
                eval_ls = s_u_eval + s_s_eval
                random.shuffle(eval_ls)
                acc_dang_ls, acc_safe_ls = [], []
                for _ in range(len(eval_ls)):
                    s_, idx = eval_ls.pop()
                    # Create dan & safe mask
                    k = min(num_obs + 1, config.TOP_K + 1)
                    dang_mask_reshape_ = np.zeros([k]).astype(bool)
                    safe_mask_reshape_ = np.ones([k]).astype(bool)
                    safe_mask_reshape_[-1] = False
                    if idx != -1:  # no dangerous h
                        dang_mask_reshape_[idx] = True
                        safe_mask_reshape_[idx] = False
                    acc_dang_, acc_safe_ = \
                        sess.run([acc_dang, acc_safe],
                                 feed_dict={s: s_, dang_mask_reshape: dang_mask_reshape_, safe_mask_reshape: safe_mask_reshape_})
                    if acc_dang_ != -1:
                        acc_dang_ls.append(acc_dang_)
                    acc_safe_ls.append(acc_safe_)
                writer.add_scalar('EVAL_ACC_DANGER', np.mean(acc_dang_ls), istep // EVAL_STEPS)
                writer.add_scalar('EVAL_ACC_SAFE', np.mean(acc_safe_ls), istep // EVAL_STEPS)

            if np.mod(istep, config.DISPLAY_STEPS) == 0:
                acc_ls = np.array(count_accuracy(acc_lists_np))
                print('Step: {}, Loss: {}, Accuracy: {}'.format(
                    istep, np.mean(loss_lists_np, axis=0),
                    acc_ls))
                writer.add_scalar('Ori_Acc_barrier_dangerous', acc_ls[0], istep // config.DISPLAY_STEPS)
                writer.add_scalar('Ori_Acc_barrier_safe', acc_ls[1], istep // config.DISPLAY_STEPS)
                loss_lists_np, acc_lists_np, dist_errors_np, safety_ratios_epoch, safety_ratios_epoch_lqr = [], [], [], [], []
        # Save CBF NN
        print("---------- Saving model ----------")
        saver_cbf.save(sess, f"{cbf_path}/model")


def train_policy(num_obs, policy, lr_cbf):
    """
    Description:
        train the policy. called by airl_model module.
    Args:
        num_obs: min(topk, num_obs)
        policy: policy
        s_: shape [batch_num, num_obs + 1, 4]
        s_next_: shape [batch_num, num_obs + 1, 4]

    Returns:

    """

    # Construct computational graph
    s, s_next, h, loss, loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv, loss_list, acc_list, num_dang, num_safe, unsafe_states = \
        build_training_graph_deriv(num_obs, policy)
    zero_ops, accumulate_ops, train_step_a = build_optimizer_deriv(loss, lr_cbf)
    tf.get_default_session().run(zero_ops)
    accumulate_ops.append(loss_list)
    accumulate_ops.append(acc_list)

    # Initialize all the uninitialized variables
    initialize_uninitialized(tf.get_default_session())

    return s, s_next, zero_ops, accumulate_ops, train_step_a, loss_list, acc_list, [num_dang, num_safe], unsafe_states, h

def train_policy_min(num_obs, policy, lr_cbf):
    """
    Description:
        train the policy. called by airl_model module.
    Args:
        num_obs: min(topk, num_obs)
        policy: policy
        s_: shape [batch_num, num_obs + 1, 4]
        s_next_: shape [batch_num, num_obs + 1, 4]

    Returns:

    """

    # Construct computational graph
    s, s_next, h, loss, acc, h_min, h_next_min, deriv_reshape = build_training_graph_deriv_min(num_obs, policy)
    zero_ops, accumulate_ops, train_step_a = build_optimizer_deriv(loss, lr_cbf)
    tf.get_default_session().run(zero_ops)

    # Initialize all the uninitialized variables
    initialize_uninitialized(tf.get_default_session())

    return s, s_next, zero_ops, accumulate_ops, train_step_a, loss, acc, h, h_min, h_next_min, deriv_reshape


# TODO
# Run pure cbf or cbf + ground truth loss_reward
def train_simple_cbf(demo_path,
                      log_path,
                      cbf_save_path,
                      policy_save_path,
                      num_obs,
                      is_load_unsafe_states=True,
                      safe_states_num=20000,
                      unsafe_states_num=20000,
                      unsafe_state_path='src/demonstrations/unsafe_states_simple_case_20000.pkl'):
    """
    Description:
        Train initial CBF NN

    Args:
        demo_path: path for demonstrations
        log_path: path for log files of tensorboard
        cbf_save_path: path for saving CBF NN
        num_obs: number of obstacles
        is_load_unsafe_states: load unsafe states or not
        unsafe_state_path: path for unsafe states

    Returns:
        None
    """


    # Tensorboard logger
    log_path = log_path
    writer = SummaryWriter(log_path)
    seed = deterministic.get_tf_seed_stream()


    # Load demonstrations
    with open(demo_path, 'rb') as f:
        demonstrations = pickle.load(f)

    # Goal state
    goal_state = demonstrations[0]['observations'][-1][-1, :]

    # Only retain topk nearest obs in demonstrations
    demonstrations = demo_remove_top_k(demonstrations, config.TOP_K)

    # Get set of states and actions from demonstrations
    S_s = [s for traj in demonstrations for s in traj['observations']]
    A_s = [a for traj in demonstrations for a in traj['actions']]

    # Collect unsafe states. Load if already exist.
    if is_load_unsafe_states:
        with open(unsafe_state_path, 'rb') as f:
            S_u, _ = pickle.load(f)
    else:
        S_u, _ = generate_simple_unsafe_states(num_obs, goal_state, unsafe_states_num)
        with open(unsafe_state_path, 'wb') as f:
            pickle.dump((S_u, _), f)

    # Add idx -1 to each s_s S_s
    S_s = [(s_s, -1) for s_s in S_s]
    # Take part of S_s
    SAFE_STATES_NUM = safe_states_num
    S_s = S_s[:SAFE_STATES_NUM]
    A_s = A_s[:SAFE_STATES_NUM]
    # Divide S_s, S_u
    traj_length = 50
    S_s_eval = S_s[len(S_s) // 2:]
    S_u_eval = S_u[len(S_u) // 2:]
    A_s_eval = A_s[len(A_s) // 2:]
    S_s = S_s[:len(S_s) // 2]
    S_u = S_u[:len(S_u) // 2]
    A_s = A_s[:len(A_s) // 2]
    TRAIN_STEPS = len(S_s) // traj_length
    unsafe_length = len(S_u) // TRAIN_STEPS
    print(">> Total training steps: ", TRAIN_STEPS)
    print(">> Training | Safe states num: ", len(S_s), ", Unsafe states num: ", len(S_u))
    print(">> Evaluation | Safe states num: ", len(S_s_eval), ", Unsafe states num: ", len(S_u_eval))

    # Start training
    print("---------- Start Training ----------")
    with tf.Session() as sess:
        # build policy
        env = GymEnv(carEnv(demo=demo_path), max_episode_length=50)
        policy = GaussianMLPPolicy(name=f'action',
                                   env_spec=env.spec,
                                   hidden_sizes=(32, 32))

        # Construct training graph
        print(goal_state)
        s, s_next, dang_mask_reshape, safe_mask_reshape, h, loss, loss_list, acc_list, loss_dang, loss_safe, acc_dang, acc_safe, loss_action, goal, a___, m =\
            build_training_graph_cbf_ground_truth(num_obs, goal_state, policy, seed)
        # step = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE).minimize(loss)
        zero_ops, accumulate_ops, train_step_h, train_step_a, grads = build_optimizer_simple(loss)
        accumulate_ops.append(loss_list)
        accumulate_ops.append(acc_list)
        # # Initialize global variables
        # sess.run(tf.global_variables_initializer())

        # Initialize all the uninitialized variables
        initialize_uninitialized(tf.get_default_session())

        # s_writer = tf.compat.v1.summary.FileWriter(log_path, sess.graph)
        # Prepare for saving CBF NN
        cbf_path = cbf_save_path
        save_dictionary_cbf = {}
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'cbf')):
            save_dictionary_cbf[f'cbf_{idx}'] = var
        saver_cbf = tf.train.Saver(save_dictionary_cbf)


        # yy: restore trained cbf
        if os.path.exists("data/simple/cbf_pure_v4/cbf"):
            saver_cbf.restore(sess, f"data/simple/cbf_pure_v4/cbf/model")







        # Prepare for saving policy NN
        save_dictionary_policy = {}
        # Add policy params
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'action')):
            save_dictionary_policy[f'action_{idx}'] = var
        saver_policy = tf.train.Saver(save_dictionary_policy)


        # Preparation
        loss_lists_np = []
        acc_lists_np = []
        # Shuffle S_s and S_u
        np.random.shuffle(S_s)
        np.random.shuffle(S_u)
        # Iterate training steps
        for istep in range(TRAIN_STEPS):
            print('>> Step: ', istep)
            # Each iteration's amount of safe and unsafe states
            S_s_iter = S_s[istep * traj_length: (istep + 1) * traj_length]
            S_u_iter = S_u[istep * unsafe_length: (istep + 1) * unsafe_length]
            A_s_iter = A_s[istep * traj_length: (istep + 1) * traj_length]
            # Re-initialize accumulators for each iteration
            sess.run(zero_ops)

            # Main training parts
            # Mix up S_s & S_u
            loss_action_tot = []
            for _ in range(len(S_s_iter) + len(S_u_iter)):
                if len(S_s_iter) == 0:
                    s_, idx = S_u_iter.pop(0)
                    s_next_ = s_
                elif len(S_u_iter) == 0:
                    s_, idx = S_s_iter.pop(0)
                    a_ = A_s_iter.pop(0)
                    s_next_ = s_ + np.concatenate([s_[:, 2:], a_], axis=1) * config.TIME_STEP  # yy: the agent row is not correct but it will be discarded in comb_core.py
                elif random.random() > .5:  # select one safe state
                    s_, idx = S_s_iter.pop(0)
                    a_ = A_s_iter.pop(0)
                    s_next_ = s_ + np.concatenate([s_[:, 2:], a_], axis=1) * config.TIME_STEP
                else:
                    s_, idx = S_u_iter.pop(0)
                    s_next_ = s_



                # Create dan & safe mask
                k = min(num_obs + 1, config.TOP_K + 1)
                dang_mask_reshape_ = np.zeros([k]).astype(bool)
                safe_mask_reshape_ = np.ones([k]).astype(bool)
                safe_mask_reshape_[-1] = False
                if idx != -1:  # no dangerous h
                    dang_mask_reshape_[idx] = True
                    safe_mask_reshape_[idx] = False

                out, loss_, loss_dang_, loss_safe_, acc_dang_, acc_safe_, loss_action_, grads_, a____, m_ =\
                    sess.run([accumulate_ops, loss, loss_dang, loss_safe, acc_dang, acc_safe, loss_action, grads, a___, m],
                             feed_dict={s: s_, s_next: s_next_, dang_mask_reshape: dang_mask_reshape_, safe_mask_reshape: safe_mask_reshape_, goal: goal_state.reshape((1, 4))})
                # print(grads_)
                # Original codebase way to add accuracy and loss
                loss_list_np, acc_list_np = out[-2], out[-1]
                loss_lists_np.append(loss_list_np)
                acc_lists_np.append(acc_list_np)
                loss_action_tot.append(loss_action_)
            print('loss_action: ', np.mean(loss_action_tot))

            writer.add_scalar('train_loss_dang_barrier', np.mean(loss_lists_np, axis=0)[0], istep)
            writer.add_scalar('train_loss_safe_barrier', np.mean(loss_lists_np, axis=0)[1], istep)
            writer.add_scalar('train_loss_dang_deriv', np.mean(loss_lists_np, axis=0)[2], istep)
            writer.add_scalar('train_loss_safe_deriv', np.mean(loss_lists_np, axis=0)[3], istep)

            # Actual Optimization
            # if np.mod(istep // 10, 2) == 0:
            #     sess.run(train_step_h)
            # else:
            #     sess.run(train_step_a)
            # sess.run(train_step_h)
            sess.run(train_step_a)

            # Evaluation
            EVAL_STEPS, EVAL_RATIO = 10, 0.5
            if np.mod(istep, EVAL_STEPS) == 0:
                s_u_eval, s_s_eval = \
                    random.sample(S_u_eval, int(len(S_u_eval) * EVAL_RATIO)), random.sample(S_s_eval, int(len(S_s_eval) * EVAL_RATIO))
                # s_u_eval, s_s_eval = [(s_u, 0) for s_u in s_u_eval], [(s_s, 1) for s_s in s_s_eval]
                eval_ls = s_u_eval + s_s_eval
                random.shuffle(eval_ls)
                acc_dang_ls, acc_safe_ls = [], []
                for _ in range(len(eval_ls)):
                    s_, idx = eval_ls.pop()
                    # Create dan & safe mask
                    k = min(num_obs + 1, config.TOP_K + 1)
                    dang_mask_reshape_ = np.zeros([k]).astype(bool)
                    safe_mask_reshape_ = np.ones([k]).astype(bool)
                    safe_mask_reshape_[-1] = False
                    if idx != -1:  # no dangerous h
                        dang_mask_reshape_[idx] = True
                        safe_mask_reshape_[idx] = False
                    acc_dang_, acc_safe_ = \
                        sess.run([acc_dang, acc_safe],
                                 feed_dict={s: s_, dang_mask_reshape: dang_mask_reshape_, safe_mask_reshape: safe_mask_reshape_})
                    if acc_dang_ != -1:
                        acc_dang_ls.append(acc_dang_)
                    acc_safe_ls.append(acc_safe_)
                writer.add_scalar('EVAL_ACC_DANGER_BARRIER', np.mean(acc_dang_ls), istep // EVAL_STEPS)
                writer.add_scalar('EVAL_ACC_SAFE_BARRIER', np.mean(acc_safe_ls), istep // EVAL_STEPS)

            if np.mod(istep, config.DISPLAY_STEPS) == 0:
                acc_ls = np.array(count_accuracy(acc_lists_np))
                print('Step: {}, Loss: {}, Accuracy: {}'.format(
                    istep, np.mean(loss_lists_np, axis=0),
                    acc_ls))
                writer.add_scalar('Ori_Acc_barrier_dangerous', acc_ls[0], istep // config.DISPLAY_STEPS)
                writer.add_scalar('Ori_Acc_barrier_safe', acc_ls[1], istep // config.DISPLAY_STEPS)
                loss_lists_np, acc_lists_np, dist_errors_np, safety_ratios_epoch, safety_ratios_epoch_lqr = [], [], [], [], []
        # Save CBF NN
        print("---------- Saving model ----------")
        saver_cbf.save(sess, f"{cbf_path}/model")
        saver_policy.save(sess, f"{policy_save_path}/model")



def train_init_CBF_NN_new(demo_path,
                      log_path,
                      cbf_save_path,
                      num_obs,
                      is_load_unsafe_states=False,
                      unsafe_ratio=1,
                      unsafe_state_path='src/demonstrations/unsafe_states_16obs_close.pkl'):
    """
    Description:
        Train initial CBF NN

    Args:
        demo_path: path for demonstrations
        log_path: path for log files of tensorboard
        cbf_save_path: path for saving CBF NN
        num_obs: number of obstacles
        is_load_unsafe_states: load unsafe states or not
        unsafe_ratio: ratio of #unsafe states to #safe states
        unsafe_state_path: path for unsafe states

    Returns:
        None
    """

    # Tensorboard logger
    log_path = log_path
    writer = SummaryWriter(log_path)

    # Load demonstrations
    with open(demo_path, 'rb') as f:
        demonstrations = pickle.load(f)

    # Only retain topk nearest obs in demonstrations
    demonstrations = demo_remove_top_k(demonstrations, config.TOP_K)

    # Goal state
    goal_state = demonstrations[0]['observations'][-1][-1, :]

    # Get set of states and actions from demonstrations
    S_s = [s for traj in demonstrations for s in traj['observations']]
    A_s = [a for traj in demonstrations for a in traj['actions']]

    # Collect unsafe states. Load if already exist.
    if is_load_unsafe_states:
        # yy: strategy 2
        S_u = list(np.load(unsafe_state_path, allow_pickle=True))
        for j, (obv, idx) in enumerate(S_u):
            topk_mask = np.argsort(np.sum(np.square((obv[:-1, :] - obv[-1, :])[:, :2]), axis=1))[:config.TOP_K]
            S_u[j] = (np.concatenate([obv[:-1, :][topk_mask, :], obv[-1, :][None, :]], axis=0), idx)

        # yy: strategy 1
        # with open(unsafe_state_path, 'rb') as f:
        #     S_u, A_u = pickle.load(f)
    else:
        # # 3 obs -> simple case
        # S_u, _ = generate_simple_unsafe_states(num_obs, goal_state, 20000)
        # with open(unsafe_state_path, 'wb') as f:
        #     pickle.dump((S_u, _), f)

        # 16 obs
        S_u, A_u = generate_unsafe_states(S_s, A_s, num_ratio=unsafe_ratio)
        with open(unsafe_state_path, 'wb') as f:
            pickle.dump([S_u, A_u], f)


    print(">> Ori total safe states num: ", len(S_s))
    print(">> Ori total unsafe states num: ", len(S_u))
    # Shuffle
    np.random.shuffle(S_s)
    np.random.shuffle(S_u)
    # # Only use parts of S_s and S_u
    S_s = S_s[:2048]
    S_u = S_u[:2048]

    # Add idx -1 to each s_s S_s
    S_s = [(s_s, -1) for s_s in S_s]
    # Divide S_s, S_u
    traj_length = 50
    S_s_eval = S_s[len(S_s) // 2:]
    S_u_eval = S_u[len(S_u) // 2:]
    S_s = S_s[:len(S_s) // 2]
    S_u = S_u[:len(S_u) // 2]
    TOTAL_EPOCH_NUM = 200
    TRAIN_STEPS = len(S_s) // traj_length
    unsafe_length = len(S_u) // TRAIN_STEPS
    print(">> Total training steps: ", TRAIN_STEPS)
    print(">> Training | Safe states num: ", len(S_s), ", Unsafe states num: ", len(S_u))
    print(">> Evaluation | Safe states num: ", len(S_s_eval), ", Unsafe states num: ", len(S_u_eval))

    # Start training
    print("---------- Start Training ----------")
    with tf.Session() as sess:
        # Construct training graph
        s, dang_mask_reshape, safe_mask_reshape, h, loss, loss_dang, loss_safe, acc_dang, acc_safe, loss_list, acc_list = build_training_graph_init(num_obs)
        zero_ops, accumulate_ops, train_step_h, train_step_a = build_optimizer(loss)
        accumulate_ops.append(loss_list)
        accumulate_ops.append(acc_list)
        # Initialize global variables
        sess.run(tf.global_variables_initializer())
        # Prepare for saving CBF NN
        cbf_path = cbf_save_path
        save_dictionary_cbf = {}
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'cbf')):
            save_dictionary_cbf[f'cbf_{idx}'] = var
        saver_cbf = tf.train.Saver(save_dictionary_cbf)

        # Preparation
        loss_lists_np = []
        acc_lists_np = []

        for m in range(1 + (TOTAL_EPOCH_NUM // TRAIN_STEPS)):
            # Iterate training steps
            for istep in range(TRAIN_STEPS):
                log_istep = m * TRAIN_STEPS + istep
                print('>> Step: ', log_istep)
                # Each iteration's amount of safe and unsafe states
                S_s_iter = S_s[istep * traj_length: (istep + 1) * traj_length]
                S_u_iter = S_u[istep * unsafe_length: (istep + 1) * unsafe_length]
                # Re-initialize accumulators for each iteration
                sess.run(zero_ops)

                # Main training parts
                # Mix up S_s & S_u
                for i in range(len(S_s_iter) + len(S_u_iter)):
                    if len(S_s_iter) == 0:
                        s_, idx = S_u_iter.pop(0)
                    elif len(S_u_iter) == 0:
                        s_, idx = S_s_iter.pop(0)
                    elif random.random() > .5:  # select one safe state
                        s_, idx = S_s_iter.pop(0)
                    else:
                        s_, idx = S_u_iter.pop(0)

                    # Create dan & safe mask
                    k = min(num_obs + 1, config.TOP_K + 1)
                    dang_mask_reshape_ = np.zeros([k]).astype(bool)
                    safe_mask_reshape_ = np.ones([k]).astype(bool)
                    safe_mask_reshape_[-1] = False
                    # yy: continue if idx is out of 12
                    if idx >= config.TOP_K:
                        continue

                    if idx != -1:  # no dangerous h
                        dang_mask_reshape_[idx] = True
                        safe_mask_reshape_[idx] = False

                    out, loss_, loss_dang_, loss_safe_, acc_dang_, acc_safe_ =\
                        sess.run([accumulate_ops, loss, loss_dang, loss_safe, acc_dang, acc_safe],
                                 feed_dict={s: s_, dang_mask_reshape: dang_mask_reshape_, safe_mask_reshape: safe_mask_reshape_})

                    # Original codebase way to add accuracy and loss
                    loss_list_np, acc_list_np = out[-2], out[-1]
                    loss_lists_np.append(loss_list_np)
                    acc_lists_np.append(acc_list_np)
                print("Step: ", log_istep, ", loss: ", np.mean(loss_lists_np, axis=0))
                writer.add_scalar('train_loss_dang', np.mean(loss_lists_np, axis=0)[0], log_istep)
                writer.add_scalar('train_loss_safe', np.mean(loss_lists_np, axis=0)[1], log_istep)
                # Actual Optimization
                sess.run(train_step_h)

                # Evaluation
                EVAL_STEPS, EVAL_RATIO = 10, 0.5
                if np.mod(log_istep, EVAL_STEPS) == 0:
                    s_u_eval, s_s_eval = \
                        random.sample(S_u_eval, int(len(S_u_eval) * EVAL_RATIO)), random.sample(S_s_eval, int(len(S_s_eval) * EVAL_RATIO))
                    # s_u_eval, s_s_eval = [(s_u, 0) for s_u in s_u_eval], [(s_s, 1) for s_s in s_s_eval]
                    eval_ls = s_u_eval + s_s_eval
                    random.shuffle(eval_ls)
                    acc_dang_ls, acc_safe_ls = [], []
                    for _ in range(len(eval_ls)):
                        s_, idx = eval_ls.pop()
                        # Create dan & safe mask
                        k = min(num_obs + 1, config.TOP_K + 1)
                        dang_mask_reshape_ = np.zeros([k]).astype(bool)
                        safe_mask_reshape_ = np.ones([k]).astype(bool)
                        safe_mask_reshape_[-1] = False
                        # yy: continue if idx is out of 12
                        if idx >= config.TOP_K:
                            continue
                        if idx != -1:  # no dangerous h
                            dang_mask_reshape_[idx] = True
                            safe_mask_reshape_[idx] = False
                        acc_dang_, acc_safe_ = \
                            sess.run([acc_dang, acc_safe],
                                     feed_dict={s: s_, dang_mask_reshape: dang_mask_reshape_, safe_mask_reshape: safe_mask_reshape_})
                        if acc_dang_ != -1:
                            acc_dang_ls.append(acc_dang_)
                        acc_safe_ls.append(acc_safe_)
                    writer.add_scalar('EVAL_ACC_DANGER', np.mean(acc_dang_ls), log_istep // EVAL_STEPS)
                    writer.add_scalar('EVAL_ACC_SAFE', np.mean(acc_safe_ls), log_istep // EVAL_STEPS)

                if np.mod(log_istep, config.DISPLAY_STEPS) == 0:
                    acc_ls = np.array(count_accuracy(acc_lists_np))
                    print('Step: {}, Loss: {}, Accuracy: {}'.format(
                        log_istep, np.mean(loss_lists_np, axis=0),
                        acc_ls))
                    writer.add_scalar('Ori_Acc_barrier_dangerous', acc_ls[0], log_istep // config.DISPLAY_STEPS)
                    writer.add_scalar('Ori_Acc_barrier_safe', acc_ls[1], log_istep // config.DISPLAY_STEPS)
                    loss_lists_np, acc_lists_np, dist_errors_np, safety_ratios_epoch, safety_ratios_epoch_lqr = [], [], [], [], []


        # Save CBF NN
        print("---------- Saving model ----------")
        saver_cbf.save(sess, f"{cbf_path}/model")



# yy: batch size based
def build_training_graph_deriv_batch(s, s_next, num_obs, policy, batch_size, is_use_two_step):
    """
    Description:
        Train policy to be safer. Freeze CBF NN and only minimizing loss_deriv.

    Args:
        s: [bs, topk + 1, 4]
        s_next: [bs, topk + 1, 4]
        num_obs: number of obstacles
        policy: the policy for the agent

    Returns:

    """
    # x is difference between the state of each agent and other agents
    x = tf.expand_dims(s, 2) - tf.expand_dims(s, 1)  # yy: shape: [bs, 13, 13, 4]
    if is_use_two_step:
        x = x[:, :, :, :2]

    # Get h() from CBF NN (h shape: 1 * 13).
    h = tf.compat.v1.map_fn(fn=lambda t: tf.reshape(comb_core.network_cbf_vel(x=t, r=config.DIST_MIN_THRES)[0], [-1]), elems=x)  # [bs, 13]

    # Get action via the policy
    a = get_action_batch(tf.reshape(s, [batch_size, 1, 4 * min(num_obs + 1, config.TOP_K + 1)]), policy, batch_size)

    # Get loss and accuracy for derivative part
    if is_use_two_step:
        loss, acc, h_min, h_next_min, deriv_reshape, deriv_safe, deriv_dang = \
            comb_core.loss_derivatives_min_batch(s, a, h, s_next, r=config.DIST_MIN_THRES, alpha=config.ALPHA_CBF)
    else:
        loss, acc, h_min, h_next_min, deriv_reshape, deriv_safe, deriv_dang = \
            comb_core.loss_derivatives_min_batch_original(s, a, h, s_next, r=config.DIST_MIN_THRES, alpha=config.ALPHA_CBF)

    # Debug ls
    debug_ls = [h_min, h_next_min, deriv_reshape, deriv_safe, deriv_dang]

    return h, loss, acc, debug_ls



if __name__ == '__main__':
    # Set seeds
    seed = 10
    deterministic.set_seed(seed)
    # # 3 obs
    # base_pth = f"data/simple/cbf_pure_v4/"
    # train_simple_cbf(demo_path="src/demonstrations/simple_3obs.pkl",
    #                   log_path=base_pth + "log",
    #                   policy_save_path=base_pth + "policy",
    #                   cbf_save_path=base_pth + "cbf",
    #                   num_obs=3)
    #
    # os.system("python scripts/airl_safe_test_simple.py --demo_path src/demonstrations/simple_3obs.pkl --policy_path " + base_pth + "policy")

    # # 3 simple NN
    # train_init_CBF_NN(demo_path="src/demonstrations/simple_3obs.pkl",
    #                   log_path="data/simple/airl_cbf_debug/log",
    #                   cbf_save_path="data/simple/airl_cbf_debug/cbf",
    #                   is_load_unsafe_states=False,
    #                   unsafe_state_path='src/demonstrations/unsafe_states_simple_case_6000.pkl',
    #                   num_obs=3)

    # 16 obs NN
    # train_init_CBF_NN(demo_path="src/demonstrations/16obs_acc_farther_target.pkl",
    #                   log_path="data/new_comb_new_demo/cbf_new_demo/log",
    #                   cbf_save_path="data/new_comb_new_demo/cbf_new_demo/cbf",
    #                   num_obs=16,
    #                   is_load_unsafe_states=False,
    #                   unsafe_state_path='src/demonstrations/unsafe_states_20000_final.pkl',)

    # # 16 obs
    # train_init_CBF_NN_new(demo_path="src/demonstrations/16obs_acc_farther_target.pkl",
    #                   log_path="data/trpo_cbf/pretrain_cbf_from_eval/log",
    #                   cbf_save_path="data/trpo_cbf/pretrain_cbf_from_eval/cbf",
    #                   num_obs=16,
    #                   is_load_unsafe_states=True,
    #                   unsafe_state_path='src/demonstrations/unsafe_states_collected_from_eval.npy',)

    # 8 obs
    train_init_CBF_NN_new(demo_path="src/demonstrations/8obs_acc_farther_target.pkl",
                      log_path="data/trpo_cbf_8obs/pretrain_cbf_2/log",
                      cbf_save_path="data/trpo_cbf_8obs/pretrain_cbf_2/cbf",
                      num_obs=8,
                      is_load_unsafe_states=False,
                      unsafe_state_path='src/demonstrations/8obs_unsafe_states.npy',)
