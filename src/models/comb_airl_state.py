import time

import tensorflow as tf
import numpy as np

from dowel import tabular
import dowel.logger as d_logger
from models.fusion_manager import RamFusionDistr
from models.imitation_learning import SingleTimestepIRL
from models.architectures import relu_net
from utils import TrainingIterator
from models.comb_cbf import *

class AIRL(SingleTimestepIRL):
    """


    Args:
        fusion (bool): Use trajectories from old iterations to train.
        state_only (bool): Fix the learned reward to only depend on state.
        score_discrim (bool): Use log D - log 1-D as reward (if true you should not need to use an entropy bonus)
        max_itrs (int): Number of training iterations to run per fit step.
    """

    def __init__(self, env,
                 expert_trajs=None,
                 reward_arch=relu_net,
                 reward_arch_args=None,
                 value_fn_arch=relu_net,
                 score_discrim=False,
                 discount=1.0,
                 state_only=False,
                 max_itrs=10,
                 fusion=False,
                 fusion_num=500,
                 name='airl_model',
                 policy=None,
                 log_path=None,
                 env_eval=None,
                 lr_cbf=1e-4,
                 start_cbf_epoch=100,
                 cbf_freq=10,
                 num_states_for_training_each_iter=256,
                 cbf_batch_size=32):
        super(AIRL, self).__init__()
        env_spec = env.spec
        if reward_arch_args is None:
            reward_arch_args = {}

        if fusion:
            self.fusion = RamFusionDistr(fusion_num, subsample_ratio=0.5)  # TODO: finetune the size
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        self.score_discrim = score_discrim
        self.gamma = discount
        assert value_fn_arch is not None
        self.set_demos(expert_trajs)
        self.state_only = state_only
        self.max_itrs = max_itrs

        self.cnt = 0  # yy: used to count the iteration nums
        self.loss_lists_np, self.acc_lists_np, self.num_lists_np = [], [], []
        # log_path = "data/comb/1_cbf_init_single/log"
        self.writer = SummaryWriter(log_path)

        # yy: For comb
        self.lr_cbf = lr_cbf
        self.start_cbf_epoch = start_cbf_epoch
        self.batch_size = cbf_batch_size
        self.num_states_for_training_each_iter = num_states_for_training_each_iter
        self.cbf_freq = cbf_freq

        # yy: old_deriv
        self.s, self.s_next, self.zero_ops, self.accumulate_ops, self.train_step_a, self.loss_list, self.acc_list, self.num_list, self.unsafe_states, self.h \
            = train_policy(num_obs=(self.dO // 4) - 1, policy=policy, lr_cbf=self.lr_cbf)
        # # yy: new_deriv
        # self.s, self.s_next, self.zero_ops, self.accumulate_ops, self.train_step_a, self.loss_deriv, self.acc_deriv, self.h, self.h_min, self.h_next_min, self.deriv_reshape \
        #     = train_policy_min(num_obs=(self.dO // 4) - 1, policy=policy, lr_cbf=self.lr_cbf)

        # yy: old_deriv
        self.accumulate_ops.append(self.loss_list)
        self.accumulate_ops.append(self.acc_list)

        self.unsafe_states_ls = []
        self.env_test = env_eval

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(tf.float32, [None, self.dO], name='obs')  # yy: current observation
            self.nobs_t = tf.placeholder(tf.float32, [None, self.dO], name='nobs')  # yy: next observation (n -> next)
            self.act_t = tf.placeholder(tf.float32, [None, self.dU], name='act')
            self.nact_t = tf.placeholder(tf.float32, [None, self.dU], name='nact')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            with tf.variable_scope('discrim') as dvs:
                rew_input = self.obs_t
                if not self.state_only:
                    rew_input = tf.concat([self.obs_t, self.act_t], axis=1)
                with tf.variable_scope('reward'):
                    self.reward = reward_arch(rew_input, dout=1, **reward_arch_args)
                    # energy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

                # value function shaping
                with tf.variable_scope('vfn'):
                    fitted_value_fn_n = value_fn_arch(self.nobs_t, dout=1)  # yy: h(s')
                with tf.variable_scope('vfn', reuse=True):
                    self.value_fn = fitted_value_fn = value_fn_arch(self.obs_t, dout=1)  # yy: h(s)

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
                self.qfn = self.reward + self.gamma * fitted_value_fn_n
                log_p_tau = self.reward + self.gamma * fitted_value_fn_n - fitted_value_fn

            log_q_tau = self.lprobs

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.discrim_output = tf.exp(log_p_tau - log_pq)  # yy: D(s, a, s')
            self.cent_loss = -tf.reduce_mean(self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq))  # yy: cross entropy loss
            self.discriminator_predict = tf.cast(log_p_tau > log_q_tau, tf.float32)
            self.discriminator_acc = tf.reduce_mean(self.discriminator_predict * self.labels +
                                                    (1 - self.discriminator_predict) * (1 - self.labels))

            self.value_loss = tf.reduce_mean(tf.losses.huber_loss(tf.stop_gradient(self.reward + self.gamma * fitted_value_fn_n), fitted_value_fn))  # yy: for updating the value function



            # self.loss = self.cent_loss + self.value_loss
            self.loss = self.cent_loss
            tot_loss = self.loss
            self.step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(tot_loss)  # TODO: tot_loss needs to be modified via combining the loss_cbf
            self._make_param_ops(_vs)





            # train_policy(policy=policy,
            #              num_obs=(self.dO // 4) - 1,
            #              istep=0,
            #              s_=np.ones((1, self.dO // 4, 4)),
            #              s_next_=np.ones((1, self.dO // 4, 4)),
            #              is_init=True)

    def fit(self, paths, policy=None, batch_size=32, logger=True, lr=1e-3, itr=0, **kwargs):
        # reward should not learn from negative noise paths
        # new_paths = []
        # for path in paths:
        #     if path['noise'][0] >= 0:
        #         new_paths.append(path)
        all_paths = paths  # yy: the reproduciblity issue comes from here, the paths is different!
        # paths = new_paths

        if self.fusion is not None:
            old_paths, _ = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths + old_paths




        # eval samples under current policy
        self._compute_path_probs(paths, insert=True)

        # eval expert log probs under current policy
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        self._insert_next_state(all_paths)
        self._insert_next_state(self.expert_trajs)
        obs, obs_next, acts, acts_next, path_probs = \
            self.extract_paths(paths,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs = \
            self.extract_paths(self.expert_trajs,
                               keys=('observations', 'observations_next', 'actions', 'actions_next', 'a_logprobs'))

        accumulated_obs, accumulated_nobs = np.ones((1, self.dO)), np.ones((1, self.dO))
        # Train discriminator
        for it in TrainingIterator(self.max_itrs, heartbeat=5):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=batch_size)  # yy: Rollouts of (s, a)

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs,
                                  batch_size=batch_size)  # yy: Expert (s, a)

            # Build feed dict
            labels = np.zeros((batch_size * 2, 1))
            labels[batch_size:] = 1.0
            # yy: AIRL's rollouted observations
            obs_batch_ori = np.copy(obs_batch)
            nobs_batch_ori = np.copy(nobs_batch)

            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            nact_batch = np.concatenate([nact_batch, nexpert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0), axis=1).astype(
                np.float32)
            feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.nact_t: nact_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr
            }
            # yy: add _ori == don't use demonstration states
            accumulated_obs = np.concatenate([accumulated_obs, obs_batch_ori], axis=0)
            accumulated_nobs = np.concatenate([accumulated_nobs, nobs_batch_ori], axis=0)





            loss, cent_loss, value_loss, _ = tf.get_default_session().run([self.loss, self.cent_loss, self.value_loss, self.step], feed_dict=feed_dict)
            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f\tCent Loss:%f\tValue Loss:%f' % (mean_loss, cent_loss, value_loss))

        start_cbf_step = self.start_cbf_epoch
        if (itr >= start_cbf_step) and (itr % self.cbf_freq == 0):

            # yy: comb part!!! Adding the CBF NN to calculate loss_safety.
            start_time = time.time()
            print("[INFO] Total data amount: ", accumulated_obs.shape[0])

            accumulated_obs, accumulated_nobs = accumulated_obs[1:, :], accumulated_nobs[1:, :]  # delete the 1st row
            train_deriv_batch_num = self.num_states_for_training_each_iter  # number of states used for calculating the derivative loss
            mask_obs = np.random.choice(accumulated_obs.shape[0], train_deriv_batch_num, replace=False)
            mask_nobs = np.random.choice(accumulated_nobs.shape[0], train_deriv_batch_num, replace=False)
            accumulated_obs = accumulated_obs[mask_obs, :]
            accumulated_nobs = accumulated_nobs[mask_nobs, :]
            s_ = np.reshape(accumulated_obs, (-1, self.dO // 4, 4))
            s_next_ = np.reshape(accumulated_nobs, (-1, self.dO // 4, 4))

            ACCUMULATE_NUM = self.batch_size

            for i in range(s_.shape[0]):
                s__, s_next__ = s_[i, :, :], s_next_[i, :, :]

                # Accumulate gradients
                # yy: old_deriv
                out, loss_list_np, acc_list_np, num_list_np, unsafe_states_, h = tf.get_default_session().run([self.accumulate_ops, self.loss_list, self.acc_list, self.num_list, self.unsafe_states, self.h], feed_dict={self.s: s__, self.s_next: s_next__})
                # # yy: new_deriv
                # out, loss, acc, h, h_min, h_next_min, deriv_reshape = tf.get_default_session().run([self.accumulate_ops, self.loss_deriv, self.acc_deriv, self.h, self.h_min, self.h_next_min, self.deriv_reshape],
                #     feed_dict={self.s: s__, self.s_next: s_next__})
                # self.loss_lists_np.append(loss)
                # self.acc_lists_np.append(acc)

                # yy: old_deriv
                if unsafe_states_ == 1:
                    self.unsafe_states_ls.append(s__)


                # Original codebase way to add accuracy and loss
                # loss_list_np, acc_list_np = out[-2], out[-1]

                # yy: old_deriv
                self.loss_lists_np.append(loss_list_np)
                self.acc_lists_np.append(acc_list_np)
                self.num_lists_np.append(num_list_np)

                # # log
                # self.writer.add_scalar('train_loss_dang_deriv', np.mean(self.loss_lists_np, axis=0)[0], itr * s_.shape[0] + i)
                # self.writer.add_scalar('train_loss_safe_deriv', np.mean(self.loss_lists_np, axis=0)[1], itr * s_.shape[0] + i)

                if self.cnt % ACCUMULATE_NUM == 0:
                    # Update gradients
                    tf.get_default_session().run(self.train_step_a)

                    # Clear accumulated gradients
                    tf.get_default_session().run(self.zero_ops)

                    # Log parts
                    # # yy: new_deriv
                    # istep = self.cnt // ACCUMULATE_NUM
                    # self.writer.add_scalar('Deriv-Loss', np.mean(self.loss_lists_np), istep)
                    # self.writer.add_scalar('Deriv-Acc', np.mean(self.acc_lists_np), istep)

                    # yy: old_deriv
                    acc_ls = np.array(count_accuracy(self.acc_lists_np))
                    istep = self.cnt // ACCUMULATE_NUM
                    print('Step: {}, Loss: {}, Accuracy: {}'.format(
                        istep, np.mean(self.loss_lists_np, axis=0), acc_ls))
                    self.writer.add_scalar('Deriv-Loss', np.mean(self.loss_lists_np), istep)
                    if acc_ls[0]:
                        self.writer.add_scalar('Ori_Acc_deriv_dangerous_filtered', acc_ls[0], istep)
                    self.writer.add_scalar('Ori_Acc_deriv_dangerous', acc_ls[0], istep)
                    self.writer.add_scalar('Ori_Acc_deriv_safe', acc_ls[1], istep)
                    if np.mean(self.loss_lists_np, axis=0)[0] > 0:
                        self.writer.add_scalar('Ori_loss_deriv_dang_filtered', np.mean(self.loss_lists_np, axis=0)[0], istep)
                    self.writer.add_scalar('Ori_loss_deriv_dang', np.mean(self.loss_lists_np, axis=0)[0], istep)
                    self.writer.add_scalar('Ori_loss_deriv_safe', np.mean(self.loss_lists_np, axis=0)[1], istep)
                    self.writer.add_scalar('Ori_num_dang', np.mean(self.num_lists_np, axis=0)[0], istep)
                    self.writer.add_scalar('Ori_num_safe', np.mean(self.num_lists_np, axis=0)[1], istep)

                    self.loss_lists_np, self.acc_lists_np, self.num_lists_np = [], [], []
                self.cnt += 1

            end_time = time.time()
            print(">> epilised time for calculating loss_deriv: ", (end_time - start_time))



        # Evaluation collision number
        eval_step = 5
        if itr % eval_step == 0:
            # Evaluation
            env_test = self.env_test
            env_test.render('no_vis')
            EVAL_TRAJ_NUM = 100

            done = False
            ob = env_test.reset()
            succ_cnt, traj_cnt, coll_cnt, cnt = 0, 0, 0, 0

            coll_ls, succ_ls = [], []
            last_timestep_state_ls = []
            while traj_cnt < EVAL_TRAJ_NUM:
                if not done:
                    ob, rew, done, info = env_test.step(policy.get_action(ob)[0])
                else:
                    # print(">> Eval traj num: ", traj_cnt)
                    traj_cnt += 1
                    coll_ls.append(info['collision_num'])
                    coll_cnt = coll_cnt + info['collision_num']
                    succ_cnt = succ_cnt + info['success']
                    last_timestep_state_ls += env_test.unsafe_states
                    ob = env_test.reset()
                    done = False

            print(">> Success traj num: ", succ_cnt, ", Collision traj num: ", coll_cnt, " out of ", EVAL_TRAJ_NUM,
                  " trajs.")

            self.writer.add_scalar('Eval - Average collision number', np.mean(coll_ls), itr)
            self.writer.add_scalar('Eval - Average success number', succ_cnt / EVAL_TRAJ_NUM, itr)
            self.writer.add_scalar('Eval - metric', (succ_cnt / EVAL_TRAJ_NUM) / np.mean(coll_ls), itr)



        if logger:
            # obs_next = np.r_[obs_next, np.expand_dims(obs_next[-1], axis=0)]
            received_learned_reward, \
            learned_value_function, \
            discrim_output, \
            discrim_acc, \
            log_cent_loss, \
            log_value_loss = \
                tf.get_default_session().run([self.reward,
                                              self.value_fn,
                                              self.discrim_output,
                                              self.discriminator_acc,
                                              self.cent_loss,
                                              self.value_loss],
                                             feed_dict={self.act_t: acts,
                                                        self.obs_t: obs,
                                                        self.nobs_t: obs_next,
                                                        self.nact_t: acts_next,
                                                        self.labels: np.zeros((acts.shape[0], 1)),
                                                        self.lprobs: np.expand_dims(path_probs, axis=1)})
            tabular.record('mean_learned_value_function', np.mean(learned_value_function))
            tabular.record('mean_received_learned_reward', np.mean(received_learned_reward))
            tabular.record('mean_log_Q', np.mean(path_probs))
            tabular.record('median_log_Q', np.median(path_probs))
            tabular.record('mean_discrim_output', np.mean(discrim_output))
            tabular.record('mean_discrim_acc', discrim_acc)
            tabular.record('loss_cent', log_cent_loss)
            tabular.record('loss_value', log_value_loss)

            # expert_obs_next = np.r_[expert_obs_next, np.expand_dims(expert_obs_next[-1], axis=0)]
            received_learned_reward, \
            learned_value_function, \
            discrim_output, \
            discrim_acc, \
            log_cent_loss, \
            log_value_loss = \
                tf.get_default_session().run([self.reward,
                                              self.value_fn,
                                              self.discrim_output,
                                              self.discriminator_acc,
                                              self.cent_loss,
                                              self.value_loss],
                                             feed_dict={self.act_t: expert_acts,
                                                        self.obs_t: expert_obs,
                                                        self.nobs_t: expert_obs_next,
                                                        self.nact_t: expert_acts_next,
                                                        self.labels: np.ones((expert_acts.shape[0], 1)),
                                                        self.lprobs: np.expand_dims(expert_probs, axis=1)})
            tabular.record('mean_learned_value_function', np.mean(learned_value_function))
            tabular.record('mean_received_learned_reward', np.mean(received_learned_reward))
            tabular.record('mean_log_Q', np.mean(expert_probs))
            tabular.record('median_log_Q', np.median(expert_probs))
            tabular.record('mean_discrim_output', np.mean(discrim_output))
            tabular.record('mean_discrim_acc', discrim_acc)
            tabular.record('loss_cent', log_cent_loss)
            tabular.record('loss_value', log_value_loss)

            # d_logger.log(tabular)

        return mean_loss

    def eval(self, paths, **kwargs):
        """
        Return bonus
        """
        if self.score_discrim:
            self._compute_path_probs(paths, insert=True)
            obs, obs_next, acts, path_probs = self.extract_paths(paths, keys=(
                'observations', 'observations_next', 'actions', 'a_logprobs'))
            path_probs = np.expand_dims(path_probs, axis=1)
            scores = tf.get_default_session().run(self.discrim_output,
                                                  feed_dict={self.act_t: acts, self.obs_t: obs,
                                                             self.nobs_t: obs_next,
                                                             self.lprobs: path_probs})
            score = np.log(scores) - np.log(1 - scores)
            score = score[:, 0]
        else:
            obs, acts = self.extract_paths(paths)
            reward = tf.get_default_session().run(self.reward,
                                                  feed_dict={self.act_t: acts, self.obs_t: obs})
            score = reward[:, 0]
        return self.unpack(score, paths)

    def eval_single(self, obs, acts):
        reward = tf.get_default_session().run(self.reward,
                                              feed_dict={self.obs_t: obs,
                                                         self.act_t: acts})
        score = reward[:, 0]
        return score

    def debug_eval(self, paths, **kwargs):
        obs, acts = self.extract_paths(paths)
        reward, v, qfn = tf.get_default_session().run([self.reward, self.value_fn,
                                                       self.qfn],
                                                      feed_dict={self.act_t: acts, self.obs_t: obs})
        return {
            'reward': reward,
            'value': v,
            'qfn': qfn,
        }
