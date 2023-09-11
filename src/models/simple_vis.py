import sys
sys.dont_write_bytecode = True

import argparse
import numpy as np


# -------------------------------- new import --------------------------------

import pickle

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from garage.tf.models import GaussianMLPModel
from mpl_toolkits.mplot3d import Axes3D

import os
from garage.experiment import deterministic
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

np.set_printoptions(4)

# MLP_MODEL_SIZE = (16, 32, 64, 64, 32, 16)
MLP_MODEL_SIZE = (32, 128, 128, 256, 256, 256, 256, 128, 128, 32)
# MLP_MODEL_SIZE = (16, 32, 64, 128, 128, 64, 32, 16)

"""
v1: r1 -> 0.2 r2 -> 0.9
v2: r1 -> 0.3 r2 -> 0.7
v3: r1 -> 0.9 r2 -> 0.95
v4: r1 -> 0.2 r2 -> 0.9 vel-x -> 0.1 pos-z -> 0.1
v5: r1 -> 0.6 r2 -> 0.9, others are back to v3
v6: r1 -> 0.4 r2 -> 0.9, others are back to v3
"""

def parse_args():
    parser = argparse.ArgumentParser()

    seed = 10
    dir = "cbf_v5"

    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=60000)
    parser.add_argument('--demo_num', type=int, default=2)

    parser.add_argument('--ratio1', type=float, default=0.4)
    parser.add_argument('--ratio2', type=float, default=0.9)
    # save paths
    parser.add_argument('--log_path', type=str, default="data/panda_final/{0}/log/".format(dir))
    parser.add_argument('--model_path', type=str, default="data/panda_final/{0}/model/".format(dir))
    parser.add_argument('--vis_path', type=str, default="data/panda_final/{0}/vis/".format(dir))
    # path to demos & unsafe states
    parser.add_argument('--demo_path', type=str, default="src/demonstrations/demos_panda_arm_push.pickle")
    parser.add_argument('--unsafe_state_path', type=str, default="src/states/states_panda_arm_push.pickle")

    args = parser.parse_args()
    return args



def CBF_NN(input, is_only_pos=False):
    """
    Input: input_state (bs, 18)
    Label: is_dang (True: dang, False: safe)
    """

    if is_only_pos:
        input = tf.concat([input[:, :3], input[:, 6:9]], axis=1)  # [bs, 6]
    else:
        # only consider ee's position and velocity, block's position. -> 9 dimensions
        input = input[:, :9]  # [bs, 9]

    g_model = GaussianMLPModel(output_dim=1, name=f'cbf', hidden_sizes=MLP_MODEL_SIZE)  # for MLP
    dist, mean, log_std = g_model.build(input).outputs
    h = mean  # [bs,]

    return h


def vis_debug():
    args = parse_args()
    # set seed
    deterministic.set_seed(args.seed)

    perspective_ls = ["up", "front", "side"]
    perspective_ls = ["paper"]
    with tf.Session() as sess:
        with open(args.unsafe_state_path, 'rb') as f:
            S_u = pickle.load(f)

        # Visualization
        # vis_vel_ls = [1e-2, 2e-2, 3e-2, 5e-2, 7e-2, 1e-1, 1.5e-1, 2e-1]
        # vis_blk_height_ls = [0.2, 0.3, 0.4, 0.5]
        # vis_vel_ls = [1e-2, 2e-2, 6e-2, 1.5e-1]
        vis_vel_ls = [1e-2]
        vis_blk_height_ls = [0.2]
        matplotlib.rcParams.update({'font.size': 15})

        s_u = S_u[np.random.choice(len(S_u), 1)[0]]
        print("s_u: ", s_u)
        s = s_u.copy()
        s = np.array([ 0, 0, 0,
                       0.07, 0.0, 0.0,
                       0.4, 0.0,  0.2,

                       1.3844e-06, 5.2096e-03,  9.7072e-03,
                       8.6520e-03,  7.5760e-05,  1.6602e-03,
                       -5.8215e-04,  3.4015e-02,  8.0817e-04])


        for perspective in perspective_ls:
            for vis_blk_height in vis_blk_height_ls:
                for vis_vel in vis_vel_ls:
                    fig = plt.figure(figsize=(15, 15))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_xlabel('x axis', fontsize=30, labelpad=20)
                    ax.set_ylabel('y axis', fontsize=30, labelpad=20)
                    ax.set_zlabel('z axis', fontsize=30, labelpad=20)

                    # # front
                    if perspective == "front":
                        ax.view_init(elev=0., azim=180)
                    # # side
                    elif perspective == "side":
                        ax.view_init(elev=0., azim=270)
                    # up and front
                    elif perspective == "up":
                        ax.view_init(elev=90., azim=180)  # ori: 5. / 230
                    else:
                        ax.view_init(elev=10., azim=260)

                    cbf_path = args.model_path

                    # s[:6], s[9:] = 0, 0
                    s[8] = vis_blk_height
                    s[3] = vis_vel

                    NUM_POINTS_EACH_AXIS = 30
                    x_range = np.arange(0, 0.65, 0.65 / NUM_POINTS_EACH_AXIS)
                    y_range = np.arange(-0.15, 0.15, 0.3 / NUM_POINTS_EACH_AXIS)
                    z_range = np.arange(0, 0.5, 0.5 / NUM_POINTS_EACH_AXIS)
                    x_ls_safe, y_ls_safe, z_ls_safe, h_ls_safe = [], [], [], []
                    x_ls_dang, y_ls_dang, z_ls_dang, h_ls_dang = [], [], [], []

                    # construct the compu graph
                    s_input = tf.compat.v1.placeholder(tf.float32, shape=(1, 18))
                    h = CBF_NN(s_input, is_only_pos=False)

                    # restore
                    save_dictionary_cbf = {}
                    for idx, var in enumerate(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'cbf')):
                        save_dictionary_cbf[f'cbf_{idx}'] = var
                    print(">> Length of save_dictionary_cbf: ", len(save_dictionary_cbf))
                    saver_cbf = tf.train.Saver(save_dictionary_cbf)
                    if os.path.exists(cbf_path):
                        print(">> CBF NN path exists!")
                        saver_cbf.restore(sess, f"{cbf_path}model")

                    for x in x_range:
                        for y in y_range:
                            for z in z_range:
                                s[0], s[1], s[2] = x, y, z
                                h_pos = sess.run(h, feed_dict={s_input: s.reshape((1, 18))})
                                if h_pos >= 0:
                                    x_ls_safe.append(x)
                                    y_ls_safe.append(y)
                                    z_ls_safe.append(z)
                                    h_ls_safe.append(h_pos)
                                else:
                                    x_ls_dang.append(x)
                                    y_ls_dang.append(y)
                                    z_ls_dang.append(z)
                                    h_ls_dang.append(h_pos)
                                if (len(h_ls_safe) + len(h_ls_dang)) % 100 == 0 and (len(h_ls_safe) + len(h_ls_dang)) > 0:
                                    print(len(h_ls_safe) + len(h_ls_dang))


                    # """
                    # safe
                    # """
                    # orig_map = plt.cm.get_cmap('Greens_r')
                    # reversed_map = orig_map.reversed()
                    # colmap = plt.cm.ScalarMappable(cmap=reversed_map)
                    # colmap.set_array(h_ls_safe)
                    # ax.scatter(x_ls_safe, y_ls_safe, z_ls_safe, marker='s', s=140, c=h_ls_safe, cmap=reversed_map,
                    #            alpha=0.1)
                    # # ax.set_zticks(fontsize=20)
                    # cb = fig.colorbar(colmap, shrink=0.5, pad=0.01)
                    # cb.ax.tick_params(labelsize=20)
                    #
                    # """
                    # dang
                    # """
                    # colmap = plt.cm.ScalarMappable(cmap=plt.cm.Reds_r)
                    # colmap.set_array(h_ls_dang)
                    # ax.scatter(x_ls_dang, y_ls_dang, z_ls_dang, marker='s', s=140, c=h_ls_dang, cmap='Reds_r', alpha=0.05)
                    # # ax.yticks(fontsize=20)
                    # cb = fig.colorbar(colmap, shrink=0.5, pad=0.01)
                    # cb.ax.tick_params(labelsize=20)


                    # orig_map = plt.cm.get_cmap('RdYlGn')
                    # reversed_map = orig_map.reversed()
                    colmap = plt.cm.ScalarMappable(cmap='RdYlGn')
                    colmap.set_array(h_ls_safe+h_ls_dang)
                    ax.scatter(x_ls_safe + x_ls_dang, y_ls_safe + y_ls_dang, z_ls_safe + z_ls_dang, marker='s', s=140,
                               c=h_ls_safe+h_ls_dang, cmap="RdYlGn", alpha=0.1)
                    # # ax.set_zticks(fontsize=20)
                    cb = fig.colorbar(colmap, shrink=0.5, pad=0.01)
                    cb.ax.tick_params(labelsize=20)


                    # draw block
                    NUM_POINTS_BLOCK_EACH_AXIS = 20
                    # x_b_range = np.arange(s_u[0] - 0.16, s_u[0] - 0.08, 0.08 / NUM_POINTS_BLOCK_EACH_AXIS)
                    x_b_range = np.arange(s_u[0] - 0.12, s_u[0] - 0.04, 0.08 / NUM_POINTS_BLOCK_EACH_AXIS)
                    # y_b_range = np.arange(-0.10, -0.07, 0.03 / NUM_POINTS_BLOCK_EACH_AXIS)
                    y_b_range = np.arange(-0.12, -0.09, 0.03 / NUM_POINTS_BLOCK_EACH_AXIS)
                    # z_b_range = np.arange(0, 0.4, 0.1 / NUM_POINTS_BLOCK_EACH_AXIS)
                    z_b_range = np.arange(0, 0.5, 0.1 / NUM_POINTS_BLOCK_EACH_AXIS)
                    x_b_ls, y_b_ls, z_b_ls = [], [], []
                    for x in x_b_range:
                        for y in y_b_range:
                            for z in z_b_range:
                                x_b_ls.append(x)
                                y_b_ls.append(y)
                                z_b_ls.append(z)
                    ax.scatter(x_b_ls, y_b_ls, z_b_ls, marker='o', s=140, color="blue", alpha=1.0)

                    # ax.tick_params(axis='both', which='major', pad=15)

                    if not os.path.exists(args.vis_path):
                        os.makedirs(args.vis_path)

                    plt.savefig(args.vis_path + '{}_vel_{}_ht_{}_total_move2.png'.format(perspective, vis_vel, vis_blk_height))




if __name__ == '__main__':
    vis_debug()
