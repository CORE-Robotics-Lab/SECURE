#!/usr/bin/env python3
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from garage.envs import GymEnv
from global_utils.utils import *
import argparse
import os
from garage.experiment import deterministic
import gym
import pickle
from torch.utils.tensorboard import SummaryWriter
from garage.tf.models import GaussianMLPModel
import panda_gym

def parse_args():
    seed = 10
    main_pth = 'panda_arm_push/bc/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_pth', type=str, default='src/demonstrations/demos_panda_arm_push.pickle')
    parser.add_argument('--save_pth', type=str, default='data/{}save'.format(main_pth))
    parser.add_argument('--log_path', type=str, default="data/{}log".format(main_pth))
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epoch_num', type=int, default=50000)
    parser.add_argument('--fusion_num', type=int, required=False, default=2000)
    parser.add_argument('--seed', type=int, required=False, default=seed)
    args = parser.parse_args()
    return args



# Get args
args = parse_args()

# Only use cpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""
config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True, log_device_placement=False)

# Set seeds
seed = args.seed
deterministic.set_seed(seed)

# Get saved paths
save_path = args.save_pth

# Load demonstrations and modify the structure
with open(args.demo_pth, 'rb') as f:
    demonstrations = pickle.load(f)

# Some params
BATCH_SIZE = 32  # should less than 50
save_dictionary = {}
writer = SummaryWriter(args.log_path)

with tf.Session(config=config) as sess:
    # Construct computational graph
    observations = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, 1, 18))
    expert_actions = tf.compat.v1.placeholder(tf.float32, shape=(BATCH_SIZE, 3))
    env = GymEnv(gym.make("PandaSafePush-v3", render=False), max_episode_length=50, is_panda=True)
    g_model = GaussianMLPModel(output_dim=3, name=f'action', hidden_sizes=[32, 32])  # for MLP
    action_dist, mean, log_std = g_model.build(observations).outputs
    learner_actions = tf.squeeze(action_dist.sample())  # [32, 3]

    loss = tf.math.reduce_mean((expert_actions - learner_actions)**2)
    step = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)
    sess.run(tf.global_variables_initializer())

    # Add policy params
    for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'action')):
        save_dictionary[f'action_{idx}'] = var

    # Start training
    for epoch in range(args.epoch_num):
        if epoch % 100 == 0:
            print("Epoch: ", epoch)
        demo_idx = np.random.choice(len(demonstrations))
        batch_idx = np.random.choice(list(range(len(demonstrations[demo_idx]['observations']))), BATCH_SIZE)
        observations_epoch = np.array(demonstrations[demo_idx]['observations'])[batch_idx, :]
        actions_epoch = np.array(demonstrations[demo_idx]['actions'])[batch_idx, :]
        loss_train, _ = sess.run([loss, step], feed_dict={observations: np.expand_dims(observations_epoch, 1), expert_actions: actions_epoch})
        writer.add_scalar('Loss', loss_train, epoch)

    # save model
    saver_share = tf.train.Saver(save_dictionary)
    saver_share.save(sess, f"{save_path}/model")



