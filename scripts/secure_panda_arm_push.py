#!/usr/bin/env python3
from garage.envs import GymEnv
from garage.tf.policies import GaussianMLPPolicy
from global_utils.utils import *
import argparse
from garage.experiment import deterministic
from models.train_cbf_nn_panda_arm_push import calc_deriv_batch, CBF_NN
import tensorflow_probability as tfp
import time
import gym
import panda_gym
import math
from models.architectures import relu_net
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def parse_args():
    parser = argparse.ArgumentParser()
    seed = 10
    parser.add_argument('--demo_path', type=str, default='src/demonstrations/demos_panda_arm_push.pickle')
    parser.add_argument('--policy_path', type=str,
                        default='data/panda_arm_push/airl_model/share')
    parser.add_argument('--airl_path', type=str,
                        default='data/panda_arm_push/airl_model/airl')
    parser.add_argument('--cbf_pth', type=str, default="data/panda_arm_push/cbf_model/model")
    parser.add_argument('--seed', type=int, required=False, default=seed)
    parser.add_argument('--alpha', type=float, required=False, default=10)
    parser.add_argument('--safe_ratio1', type=float, required=False, default=0.1)
    parser.add_argument('--safe_ratio2', type=float, required=False, default=0.3)
    parser.add_argument('--method', type=int, required=False, default=1)
    args = parser.parse_args()
    return args


def fibonacci_sphere(samples=1000):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append([x, y, z])

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


def sort_safe_actions(q_graph, s_ph, s_next_ph, is_safe_graph, is_safe, actions, s, s_next, safe_ratio, method, sess):
    bs = s.shape[0]
    q = sess.run(q_graph, feed_dict={s_ph: s.reshape((-1, 18)),
                                     s_next_ph: s_next.reshape((-1, 18))}).reshape([-1])
    safe_idxes = np.where(is_safe)[0]
    q_safe = q[safe_idxes]
    safe_actions = actions[safe_idxes, :]

    # get data to compare and analyse

    # m1 - pure avg
    action_m1 = np.mean(actions[safe_idxes, :], axis=0)

    # m2 - avg of 70% safe actions
    sorted_q_safe = np.argsort(q_safe)[::-1]
    action_m2 = np.mean(actions[safe_idxes[sorted_q_safe[-int(sorted_q_safe.shape[0]*safe_ratio):]], :], axis=0)

    # m3 - pure most similar one to avg 70%
    norm_safe_actions = np.linalg.norm(safe_actions, axis=1)
    norm_avg = np.linalg.norm(action_m2, axis=0)
    similarity = np.dot(safe_actions, action_m2) / (norm_safe_actions * norm_avg)
    action_m3 = safe_actions[np.argmax(similarity), :]

    # m4 - safe action with highest q value
    action_m4 = actions[safe_idxes[np.argmax(q_safe)], :]

    # m5 - avg of 70% safe actions with safety check + most similar one if it's unsafe
    is_avg_safe = sess.run(is_safe_graph, feed_dict={
        s_ph: np.tile(s[0, :], bs).reshape((-1, 18)),
        s_next_ph: np.tile(dynamic_model(s[0, :], action_m2.reshape((-1, 3)), 1)[1], bs).reshape((-1, 18))})[0][0]
    if is_avg_safe:
        action_m5 = action_m2.copy()
    else:
        action_m5 = action_m3.copy()

    # combination of all methods to compare
    comb_comp = (action_m1, action_m2, action_m3, action_m4, action_m5)

    return comb_comp[method-1], safe_actions.shape[0] == 0, comb_comp


# Global params
EVAL_TRAJ_NUM = 100
VIDEO_TRAJ_NUM = 20
INCREASE_VAR_LINE = 50
INCREASE_INTERVAL = 50
INCREASE_RATIO = 0.05
MAX_SAFE_WAIT_TIME = 100
BATCH_SIZE = 128
SAMPLE_NUM=1000
SPHERE_POINTS = fibonacci_sphere(samples=SAMPLE_NUM)
RESAMPLE_NUM_NO_ADAPTIVE = 10

# Get params
args = parse_args()
log_path = args.policy_path
demo_pth = args.demo_path

# Set seeds
seed = args.seed
deterministic.set_seed(seed)

# Create env for graph construction
env = GymEnv(gym.make("PandaSafePush-v3", render=False), max_episode_length=50, is_panda=True)

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
    env_test = gym.make("PandaSafePush-v3", render=True)
    ob = env_test.reset()

    # Preparation
    imgs, coll_ls, succ_ls = [], [], []
    done, is_safe = False, False
    succ_cnt, traj_cnt, fall_cnt, cnt = 0, 0, 0, 0

    # Construct graph
    s_around = tf.placeholder(tf.float32, [SAMPLE_NUM, 9], name='states_around')
    h_states_around = CBF_NN(s_around, is_only_pos=False)
    # is_safe_action, ph, deriv = calc_deriv_batch(config_cbf.ALPHA_CBF, BATCH_SIZE, is_only_pos=False)
    is_safe_action, ph, deriv = calc_deriv_batch(args.alpha, BATCH_SIZE, is_only_pos=False)  # use 10 for alpha (todo)
    s, s_next = ph

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
    loc, scale_diag = tf.placeholder(tf.float32, [3,], name='loc'), tf.placeholder(tf.float32, [3,], name='scale_diag')
    dist = tfp.distributions.MultivariateNormalDiag(
        loc=loc,
        scale_diag=scale_diag)
    samples = dist.sample([BATCH_SIZE])


    # load q function model
    gamma = 0.99
    with tf.variable_scope('reward'):
        reward = relu_net(s, dout=1)
    with tf.variable_scope('vfn'):
        fitted_value_fn_n = relu_net(s_next, dout=1)
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

                # where do the model transition
                ob_batch, ob_next_batch = dynamic_model(ob, action, bs=BATCH_SIZE)

                # for debug
                crr_ee_pos, crr_ee_vel, crr_blk_pos, crr_blk_vel = ob[:3], ob[3:6], ob[6:9], ob[9:12]
                nxt_ee_pos, nxt_ee_vel, nxt_blk_pos, nxt_blk_vel = ob_next_batch[0, :3], ob_next_batch[0, 3:6], \
                                                                   ob_next_batch[0, 6:9], ob_next_batch[0, 9:12]

                # get is_safe by using ob & ob_next
                get_is_safe_tm_s = time.time()
                is_safe, deriv_ = sess.run([is_safe_action, deriv], feed_dict={
                    s: ob_batch,
                    s_next: ob_next_batch})
                get_is_safe_tm_e = time.time()
                get_is_safe_tm_ls.append(get_is_safe_tm_e - get_is_safe_tm_s)

                if cnt > MAX_SAFE_WAIT_TIME:
                    action = action[0, :]
                    break

                if np.sum(is_safe) >= int(args.safe_ratio1 * BATCH_SIZE):
                    action, no_forward_action, comb_comp = sort_safe_actions(q_graph, s, s_next, is_safe_action,
                                               is_safe, action, ob_batch, ob_next_batch, args.safe_ratio2, args.method, sess)

                    a_comp_ls.append(comb_comp)

                    break
                elif cnt < RESAMPLE_NUM_NO_ADAPTIVE:
                    action = sess.run([samples], feed_dict={loc: mean, scale_diag: std_init})[0].reshape(BATCH_SIZE, -1)
                else:
                    # Detect which direction to increase std
                    R = 0.2
                    ee_pos = ob[:3]
                    # move the sphere to the ee_pos
                    around_points = np.array(SPHERE_POINTS) * R + np.tile(ee_pos, (len(SPHERE_POINTS), 1))  # [SAMPLE_NUM, 3]
                    # create states for points on the sphere
                    around_states = np.concatenate([around_points, np.tile(ob[3:9], (len(SPHERE_POINTS), 1))], axis=1)  # [SAMPLE_NUM, 9]
                    # get h values for points on the sphere
                    around_states_h = sess.run(h_states_around, feed_dict={s_around: around_states}).reshape((SAMPLE_NUM, 1))  # [SAMPLE_NUM, 1]
                    # filter -> we only need the points with negative h value
                    coef = np.max(np.concatenate([-1 * around_states_h, np.zeros_like(around_states_h)], axis=1), axis=1, keepdims=True)  # [SAMPLE_NUM, 1]
                    # get points on sphere without transition
                    around_points_no_transition = np.array(SPHERE_POINTS) * R
                    # get std delta
                    std_delta = np.abs(np.sum((coef[:, None] * around_points_no_transition[..., None]).reshape(around_points_no_transition.shape[0], -1), axis=0))  # [3])
                    # normalize std delta
                    std_delta /= np.max(std_delta)
                    # this means all points on the sphere have positive h value
                    if np.all(coef == 0):
                        mean, std = mean, std_init + np.array([1, 1, 1]) * 0.01 * (cnt - RESAMPLE_NUM_NO_ADAPTIVE)
                    else:
                        # std_delta[1] = 0
                        mean, std = mean, std_init + std_delta * 0.1 * (cnt - RESAMPLE_NUM_NO_ADAPTIVE)
                    action = sess.run([samples], feed_dict={loc: mean, scale_diag: std})[0].reshape(BATCH_SIZE, -1)

            resample_times.append(cnt)

            # if it's safe action, break the while loop and continue
            traj_data["actions"].append(action)
            ob, rew, done, _, info = env_test.step(action)
            fall_angle = max(info['fall_angle'], fall_angle)
            if info['is_fall']:
                is_fall = True
            is_safe = False

            get_safe_action_avg_ls.append(time.time() - tp)

            if traj_cnt < VIDEO_TRAJ_NUM:
                imgs.append(env_test.render('rgb_array')[:, :, :3])
        else:
            traj_data["observations"] = np.array(traj_data["observations"])
            traj_data["actions"] = np.array(traj_data["actions"])
            traj_datas.append(traj_data)
            traj_data = {"observations": [], "actions": []}
            new_timepoint = time.time()
            eval_traj_time.append(new_timepoint - timepoint)
            timepoint = new_timepoint
            print(">> Eval traj num: ", traj_cnt)
            traj_cnt += 1
            if not is_fall:
                succ_cnt = succ_cnt + info['is_success']
            print(">> is success: ", info['is_success'])
            fall_cnt = fall_cnt + is_fall
            print(">> is fall: ", is_fall)
            print(">> Fall angle: ", fall_angle)
            fall_angle_ls.append(fall_angle)
            ob = env_test.reset()
            done = False
            is_fall = False
            fall_angle = 0

    # Log info
    print(">> Success traj num: ", succ_cnt, ", fall num: ", fall_cnt, " out of ", EVAL_TRAJ_NUM, " trajs.")
    print(succ_ls)
    print("avg calc is_safe time:", np.mean(get_is_safe_tm_ls))
    print("avg resample times:", np.mean(resample_times))
    print("avg traj time:", np.mean(eval_traj_time), len(eval_traj_time), eval_traj_time)
    print("get_safe_action_avg_ls: ", np.mean(get_safe_action_avg_ls), len(get_safe_action_avg_ls))
    print("fall_angle_ls: ", np.mean(fall_angle_ls), len(fall_angle_ls), fall_angle_ls)
    save_video(imgs, log_path + "/secure.avi")

    # Write log
    with open(log_path + "/eval_results_secure.txt", 'w', encoding='utf-8') as f:
        f.write(">> Success traj num: " + str(succ_cnt) + ", fall num: " + str(fall_cnt) + " out of " + str(EVAL_TRAJ_NUM) + " trajs.\n")
        # f.write(str(fall_angle_ls))
        f.write("\n>> Fall angle avg: " + str(np.mean(fall_angle_ls)))

