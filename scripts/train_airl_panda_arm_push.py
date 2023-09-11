#!/usr/bin/env python3
# import tensorflow as tf
import tensorflow.compat.v1 as tf
from garage.envs import GymEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from airl.irl_trpo import TRPO
from models.airl_state import AIRL
from garage.tf.policies import GaussianMLPPolicy
from garage.trainer import Trainer
from global_utils.utils import *
from garage.experiment import Snapshotter
import dowel
from dowel import logger
import argparse
import os
from garage.experiment import deterministic
import gym
import pickle
import panda_gym

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    seed = 10
    main_pth = 'data/panda_arm_push/airl_model'

    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_path', type=str, default='data/panda_arm_push/bc/save')
    # For AIRL training - hyperparameters
    parser.add_argument('--demo_num', type=int, required=False, default=2)
    parser.add_argument('--epoch_num', type=int, required=False, default=100)

    parser.add_argument('--fusion_num', type=int, required=False, default=2000)
    parser.add_argument('--max_itrs', type=int, required=False, default=10)
    parser.add_argument('--max_kl_step', type=float, required=False, default=0.0001)
    parser.add_argument('--generator_train_itrs', type=int, required=False, default=10)
    parser.add_argument('--discrim_train_itrs', type=int, required=False, default=10)

    parser.add_argument('--share_pth', type=str, default=main_pth + "/share")
    parser.add_argument('--airl_pth', type=str, default=main_pth + "/airl_model")
    parser.add_argument('--demo_pth', type=str, default='src/demonstrations/demos_panda_arm_push.pickle')
    parser.add_argument('--is_bc', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, required=False, default=seed)
    args = parser.parse_args()
    return args


def demo_remove_top_k(demos, topk):
    for i, demo in enumerate(demos):
        obvs = demo['observations']
        for j, obv in enumerate(obvs):
            topk_mask = np.argsort(np.sum(np.square((obv[:-1, :] - obv[-1, :])[:, :2]), axis=1))[:topk]
            demos[i]['observations'][j] = np.concatenate([obv[:-1, :][topk_mask, :], obv[-1, :][None, :]], axis=0)
    return demos

args = parse_args()

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
config = tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True, log_device_placement=False)

# log_path = args.log_pth
share_path = args.share_pth
airl_path = args.airl_pth

# Set seeds
seed = args.seed
deterministic.set_seed(seed)

# params
NUM_DEMO_USED = args.demo_num
EPOCH_NUM = args.epoch_num
demo_pth = args.demo_pth

irl_models = []
policies = []
algos = []
trainers = []

# Load demonstrations and create environment
with open(demo_pth, 'rb') as f:
    demonstrations = pickle.load(f)

demonstrations = [demonstrations[:NUM_DEMO_USED]]
env = GymEnv(gym.make("PandaSafePush-v3", render=False), max_episode_length=50, is_panda=True)

with tf.Session(config=config) as sess:
    save_dictionary_share = {}
    save_dictionary_airl = {}
    save_dictionary = {}
    for index in range(len(demonstrations)):
        snapshotter = Snapshotter(f'{share_path}/skill_{index}')
        trainer = Trainer(snapshotter)

        # AIRL
        irl_model = AIRL(env=env, expert_trajs=demonstrations[index],
                         state_only=True, fusion=True,
                         max_itrs=args.max_itrs,
                         name=f'skill_{index}',
                         fusion_num=args.fusion_num,
                         is_collect_rollout_states=False)

        # policy
        policy = GaussianMLPPolicy(name=f'action',
                                   env_spec=env.spec,
                                   hidden_sizes=(32, 32))

        # Add airl_model params
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'skill_{index}')):
            save_dictionary_airl[f'my_skill_{index}_{idx}'] = var

        # Add policy params
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'action')):
            save_dictionary[f'action_{idx}'] = var
            save_dictionary_share[f'action_{idx}'] = var
            save_dictionary_airl[f'action_{idx}'] = var

        # Add reward params
        for idx, var in enumerate(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                              scope=f'skill_{index}/discrim/reward')):
            save_dictionary_share[f'reward_{index}_{idx}'] = var

        # restore policy and airl_model
        if os.path.exists(airl_path):
            saver = tf.train.Saver(save_dictionary_airl)
            saver.restore(sess, f"{airl_path}/model")

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = None

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    index=index,
                    sampler=sampler,
                    irl_model=irl_model,
                    generator_train_itrs=args.generator_train_itrs,
                    discrim_train_itrs=args.discrim_train_itrs,
                    policy_ent_coeff=0.0,
                    discount=0.99,
                    max_kl_step=args.max_kl_step)
        trainers.append(trainer)
        irl_models.append(irl_model)
        policies.append(policy)
        algos.append(algo)

    sess.run(tf.global_variables_initializer())

    # if bc warm up, restore
    if args.is_bc:
        print(">> Load BC pre-trained model")
        if os.path.exists(args.bc_path):
            print(">> bc model path exists")
        saver = tf.train.Saver(save_dictionary)
        saver.restore(sess, f"{args.bc_path}/model")

    for i in range(len(demonstrations)):
        # Training
        trainer = trainers[i]

        sampler = LocalSampler(agents=policies[i],
                               envs=[env],
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               n_workers=1)

        algos[i]._sampler = sampler

        logger.remove_all()
        logger.add_output(dowel.StdOutput())
        logger.add_output(dowel.TensorBoardOutput(f"{share_path}/policy_logs/"))
        logger.log('Starting up...')

        trainer.setup(algos[i], env)
        trainer.train(n_epochs=EPOCH_NUM, batch_size=200)  # 10000

        # save model
        saver_share = tf.train.Saver(save_dictionary_share)
        saver_share.save(sess, f"{share_path}/model")

        saver_airl = tf.train.Saver(save_dictionary_airl)
        saver_airl.save(sess, f"{airl_path}/model")