import sys
sys.dont_write_bytecode = True
import argparse
import numpy as np
import models.config as config
from garage.tf.models import GaussianMLPModel, GaussianCNNModel, GaussianConv1Model
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from torch.utils.tensorboard import SummaryWriter
import os
from garage.experiment import deterministic

np.set_printoptions(4)

MLP_MODEL_SIZE = (32, 128, 128, 256, 256, 256, 256, 128, 128, 32)

def parse_args():
    parser = argparse.ArgumentParser()

    seed = 10
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch_num', type=int, default=60000)
    parser.add_argument('--demo_num', type=int, default=2)

    parser.add_argument('--ratio1', type=float, default=0.3)
    parser.add_argument('--ratio2', type=float, default=0.7)
    # save paths
    parser.add_argument('--log_path', type=str, default="data/panda_arm_push/cbf_model/log/")
    parser.add_argument('--model_path', type=str, default="data/panda_arm_push/cbf_model/model/")
    parser.add_argument('--vis_path', type=str, default="data/panda_arm_push/cbf_model/vis/")
    # path to demos & unsafe states
    parser.add_argument('--demo_path', type=str, default="src/demonstrations/demos_panda_arm_push.pickle")
    parser.add_argument('--unsafe_state_path', type=str, default="src/states/states_panda_arm_push.pickle")

    args = parser.parse_args()
    return args


# The following 2 functions are used by another file for secure

# Used by calc_deriv_batch
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

# Used by secure to calculate the derivative of CBF in order to get is_safe
def calc_deriv_batch(alpha, batch_size, is_only_pos=False):
    # placeholders
    s = tf.placeholder(tf.float32, [batch_size, 18], name='state')
    s_next = tf.placeholder(tf.float32, [batch_size, 18], name='state_next')
    ph = [s, s_next]

    # calc h & h_next
    h = CBF_NN(s, is_only_pos)  # [bs,]
    h_next = CBF_NN(s_next, is_only_pos)  # [bs,]

    # calc deriv
    deriv = h_next - h + config.TIME_STEP * alpha * h  # [bs,]

    # True if safe, False if dangerous
    is_safe_action = tf.math.less_equal(tf.zeros_like(deriv, dtype=tf.float32), deriv)  # [bs,]
    return is_safe_action, ph, deriv


# yy -------------------------------- CBF NN related functions --------------------------------

def generate_unsafe_states_final(demos, r1=0.3, r2=0.8):
    RATIO_SAFEVEL_UNSAFEPOS = r1
    RATIO_UNSAFEVEL_SAFEPOS = r2
    UNSAFE_VEL_RANGE = np.arange(3.0e-2, 2.0e-1, 5e-4)
    POS_Z_THRESHOLD = 0.07

    unsafe_states_all = np.zeros([1, 18])
    for demo in demos:
        states = np.array([s for s in demo['observations']])
        actions = np.array(demo['actions'])
        # find the index of the first row in actions whose third dimension's absolute value is less than 0.3.
        idx = np.argmax(np.logical_and(np.abs(actions[:, 0]) < 0.3, actions[:, 2] == 0))
        for _ in range(10):
            states_dang = states[idx:, :].copy()
            random_val = np.random.rand()
            # safe velocity, unsafe position
            if random_val < RATIO_SAFEVEL_UNSAFEPOS:
                states_dang[:, 0] -= 0.02e-1
                states_dang[:, 2] = np.random.choice(np.arange(POS_Z_THRESHOLD, 0.4, 1e-3), states_dang.shape[0]).T

            # unsafe velocity, safe position
            # 1. push fast at x axis
            # 2. push fast at z axis
            elif random_val < RATIO_UNSAFEVEL_SAFEPOS:
                random_val_2 = np.random.rand()
                # push fast at x axis
                if random_val_2 < 0.6:

                    # push position can vary in unsafe position range
                    states_dang[:, 0] -= 0.02e-1
                    states_dang[:, 2] = np.random.choice(np.arange(0, POS_Z_THRESHOLD, 1e-3), states_dang.shape[0]).T
                    states_dang[:, 3] = np.random.choice(UNSAFE_VEL_RANGE, states_dang.shape[0]).T

                # push fast at z axis
                elif random_val_2 < 0.8:
                    # push position can vary in unsafe position range
                    states_dang[:, 0] -= 0.02e-1
                    states_dang[:, 2] = np.random.choice(np.arange(0, POS_Z_THRESHOLD, 1e-3), states_dang.shape[0]).T
                    states_dang[:, 5] = np.random.choice(np.arange(1e-2, 2.0e-1, 5e-4), states_dang.shape[0]).T

            # unsafe velocity, unsafe position
            else:
                height_block = states[0, 8] * 2
                states_dang[:, 0] -= 0.02e-1
                states_dang[:, 2] = np.random.choice(np.arange(POS_Z_THRESHOLD, 0.4, 1e-3), states_dang.shape[0]).T
                states_dang[:, 3] = np.random.choice(UNSAFE_VEL_RANGE, states_dang.shape[0]).T

            unsafe_states_all = np.vstack([unsafe_states_all, states_dang])

    # add some noise to y axis
    add_y = unsafe_states_all.copy()
    add_y[1:, 1] += np.random.uniform(-0.001, 0.001, unsafe_states_all.shape[0] - 1)
    unsafe_states_all = np.vstack([unsafe_states_all, add_y])
    return list(unsafe_states_all[1:, :])


def safe_states_augmentation(S_s):
    augmented_S_s_ls = []
    AUGMENTED_NUM = 10  # ori: 10
    for S_s_ in S_s:
        for _ in range(AUGMENTED_NUM):
            ss = S_s_.copy()
            # pos_x
            ss[0] += np.random.uniform(-0.01, 0)
            # pos_y
            ss[1] += np.random.uniform(-0.005, 0.005)
            # pos_z
            ss[2] += np.random.uniform(-0.001, 0.001)
            # vel_x
            ss[3] += np.random.uniform(-0.001, 0.001)
            # vel_y
            ss[4] += np.random.uniform(-0.001, 0.001)
            # vel_z
            ss[5] += np.random.uniform(-0.001, 0.001)
            augmented_S_s_ls.append(ss)
    return augmented_S_s_ls



def build_Gaussian_NN(input, label, is_only_pos=False):
    """
    Input: input_state (bs, 18)
    Label: is_dang (True: dang, False: safe)
    """

    if is_only_pos:
        input = tf.concat([input[:, :3], input[:, 6:9]], axis=1)  # [bs, 6]
    else:
        # only consider ee's position and velocity, block's position. -> 9 dimensions
        input = input[:, :9]

    g_model = GaussianMLPModel(output_dim=1, name=f'cbf', hidden_sizes=MLP_MODEL_SIZE)
    dist, mean, log_std = g_model.build(input).outputs
    h = mean

    # Formulate the loss_barrier
    loss_barrier = tf.cond(label,
                           lambda: tf.math.maximum(h + 1e-3, 0),
                           lambda: tf.math.maximum(-h + 5e-2, 0))

    return dist, h, log_std, loss_barrier


def NN_AL(is_load_unsafe_states=False, is_only_pos=False):
    args = parse_args()

    # set seed
    deterministic.set_seed(args.seed)

    # Tensorboard logger
    writer = SummaryWriter(args.log_path)

    # Load demonstrations
    with open(args.demo_path, 'rb') as f:
        demonstrations = pickle.load(f)

    # number of demonstrations used
    demonstrations = demonstrations[:args.demo_num]

    # Get set of states and actions from demonstrations

    # Get safe states and augment them with random noise
    S_s = [s for traj in demonstrations for s in traj['observations']]
    S_s = safe_states_augmentation(S_s)

    # Get unsafe states.
    # Load if already exist.
    if is_load_unsafe_states:
        with open(args.unsafe_state_path, 'rb') as f:
            S_u = pickle.load(f)
    else:
        S_u = generate_unsafe_states_final(demonstrations, args.ratio1, args.ratio2)
        with open(args.unsafe_state_path, 'wb') as f:
            pickle.dump(S_u, f)

    # Shuffle S_s and S_u
    np.random.shuffle(S_s)
    np.random.shuffle(S_u)

    # Split dataset into train, eval, test
    dataset_size = min(len(S_s), len(S_u))
    train_size = int(dataset_size * 0.6)
    eval_size = int(dataset_size * 0.2)
    S_s_train = S_s[:train_size]
    S_s_eval = S_s[train_size:train_size + eval_size]
    S_s_test = S_s[train_size + eval_size:]
    S_u_train = S_u[:train_size]
    S_u_eval = S_u[train_size:train_size + eval_size]
    S_u_test = S_u[train_size + eval_size:]

    print(">> Total safe states num: {}, Train num: {}, Eval num: {}, Test num: {}"
          .format(len(S_s), len(S_s_train), len(S_s_eval), len(S_s_test)))
    print(">> Total unsafe states num: {}, Train num: {}, Eval num: {}, Test num: {}"
          .format(len(S_u), len(S_u_train), len(S_u_eval), len(S_u_test)))


    with tf.Session() as sess:
        # Build training graph
        safe_num, dang_num = args.batch_size // 2, args.batch_size - (args.batch_size // 2)
        state_input_safe = tf.compat.v1.placeholder(tf.float32, shape=(safe_num, 18))
        state_input_dang = tf.compat.v1.placeholder(tf.float32, shape=(dang_num, 18))
        _, h_safe, _, loss_barrier_safe = \
            build_Gaussian_NN(state_input_safe, tf.constant(False, dtype=tf.bool), is_only_pos=is_only_pos)
        _, h_dang, _, loss_barrier_dang = \
            build_Gaussian_NN(state_input_dang, tf.constant(True, dtype=tf.bool), is_only_pos=is_only_pos)

        # Give diff weight for dang/safe cases
        loss_list = [loss_barrier_dang, loss_barrier_safe]
        loss = 10 * tf.math.add_n(loss_list)
        step = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)
        sess.run(tf.global_variables_initializer())

        # Save model
        save_dictionary_cbf = {}
        for idx, var in enumerate(
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                  scope=f'cbf')):
            save_dictionary_cbf[f'cbf_{idx}'] = var
        saver_cbf = tf.train.Saver(save_dictionary_cbf)

        # Start training
        print("---------- Start Training ----------")
        for epoch in range(args.epoch_num):
            print(">> Epoch: ", epoch) if epoch % 10 == 0 else None
            # Get one batch
            s_s_batch_train = np.array(S_s_train)[np.random.choice(len(S_s_train), size=safe_num)]
            s_u_batch_train = np.array(S_u_train)[np.random.choice(len(S_u_train), size=dang_num)]

            # Execute GD
            _, loss_total, loss_barrier_safe_epoch, loss_barrier_dang_epoch = \
                sess.run([step, loss, loss_barrier_safe, loss_barrier_dang],
                         feed_dict={state_input_safe: s_s_batch_train, state_input_dang: s_u_batch_train})

            # Calculate training accuracy
            h_safe_ls = sess.run(h_safe, feed_dict={state_input_safe: s_s_batch_train})
            h_dang_ls = sess.run(h_dang, feed_dict={state_input_dang: s_u_batch_train})

            safe_pred = h_safe_ls >= 0
            dang_pred = h_dang_ls < 0
            accuracy = np.mean(list(safe_pred) + list(dang_pred))

            # Log
            writer.add_scalar('Accuracy_Train', accuracy, epoch)
            writer.add_scalar('loss_barrier', np.mean(loss_total), epoch)
            writer.add_scalar('loss_safe', np.mean(loss_barrier_safe_epoch), epoch)
            writer.add_scalar('loss_dang', np.mean(loss_barrier_dang_epoch), epoch)

            # Evaluation
            if epoch % 5 == 0:
                eval_acc_ls = []
                for _ in range(10):
                    # Get one batch
                    s_s_eval_train = np.array(S_s_eval)[np.random.choice(len(S_s_eval), size=safe_num), :]
                    s_u_eval_train = np.array(S_u_eval)[np.random.choice(len(S_u_eval), size=dang_num), :]

                    # Calculate evaluation accuracy
                    safe_pred_eval = sess.run(h_safe, feed_dict={state_input_safe: s_s_eval_train}) >= 0
                    dang_pred_eval = sess.run(h_dang, feed_dict={state_input_dang: s_u_eval_train}) < 0
                    eval_accuracy = np.mean(list(safe_pred_eval) + list(dang_pred_eval))

                    eval_acc_ls.append(eval_accuracy)

                # Log
                writer.add_scalar('Accuracy_EVAL', np.mean(eval_acc_ls), epoch)


        print("---------- Saving model ----------")
        # Save CBF NN
        saver_cbf.save(sess, f"{args.model_path}/model")

        # Test time
        # Calculate evaluation accuracy
        test_acc_ls = []
        for _ in range(1000):
            s_s_test_train = np.array(S_s_test)[np.random.choice(len(S_s_test), size=safe_num), :]
            s_u_test_train = np.array(S_u_test)[np.random.choice(len(S_u_test), size=dang_num), :]
            safe_pred_test = np.min(sess.run(h_safe, feed_dict={state_input_safe: s_s_test_train}), axis=1) >= 0
            dang_pred_test = np.min(sess.run(h_dang, feed_dict={state_input_dang: s_u_test_train}), axis=1) < 0
            test_accuracy = np.mean(list(safe_pred_test) + list(dang_pred_test))
            test_acc_ls.append(test_accuracy)
        print(">> Test accuracy: ", np.mean(test_acc_ls))
        with open(os.path.join(args.model_path, "test.txt"), "a") as f:
            f.writelines(str(np.mean(test_acc_ls)))


if __name__ == '__main__':
    NN_AL(is_load_unsafe_states=True)