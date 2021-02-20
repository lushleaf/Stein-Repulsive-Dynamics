# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple example of contextual bandits simulation.

Code corresponding to:
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks
for Thompson Sampling, by Carlos Riquelme, George Tucker, and Jasper Snoek.
https://arxiv.org/abs/1802.09127
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import app
from absl import flags
import numpy as np
import os
import datetime
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.core.contextual_bandit import run_contextual_bandit
from bandits.data.data_sampler import sample_adult_data
from bandits.data.data_sampler import sample_census_data
from bandits.data.data_sampler import sample_covertype_data
from bandits.data.data_sampler import sample_jester_data
from bandits.data.data_sampler import sample_mushroom_data
from bandits.data.data_sampler import sample_statlog_data
from bandits.data.data_sampler import sample_stock_data
from bandits.algorithms.fixed_policy_sampling import FixedPolicySampling
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.data.synthetic_data_sampler import sample_sparse_linear_data
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data
from bandits.algorithms.uniform_sampling import UniformSampling

# Set up your file routes to the data files.
base_route = os.getcwd()
data_route = 'data'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('bandit', 'mushroom', 'which game do you want?')
flags.DEFINE_string('layers', '100,100', 'which game do you want?')
flags.DEFINE_integer('interp_batch_size', 64, 'which game do you want?')
flags.DEFINE_float('mm_jitter', 1e-4, 'which game do you want?')
flags.DEFINE_float('lr', 1e-4, 'which game do you want?')
flags.DEFINE_float('prior_variance', 10, 'which game do you want?')
flags.DEFINE_integer('num_context', 50000, '=')
flags.DEFINE_integer('seed', 1234, '=')
flags.DEFINE_string('logdir', './bandits/results/', 'Base directory to save output')
flags.DEFINE_string(
    'mushroom_data',
    os.path.join(base_route, data_route, 'mushroom.data'),
    'Directory where Mushroom data is stored.')
flags.DEFINE_string(
    'financial_data',
    os.path.join(base_route, data_route, 'raw_stock_contexts'),
    'Directory where Financial data is stored.')
flags.DEFINE_string(
    'jester_data',
    os.path.join(base_route, data_route, 'jester_data_40jokes_19181users.npy'),
    'Directory where Jester data is stored.')
flags.DEFINE_string(
    'statlog_data',
    os.path.join(base_route, data_route, 'shuttle.trn'),
    'Directory where Statlog data is stored.')
flags.DEFINE_string(
    'adult_data',
    os.path.join(base_route, data_route, 'adult.full'),
    'Directory where Adult data is stored.')
flags.DEFINE_string(
    'covertype_data',
    os.path.join(base_route, data_route, 'covtype.data'),
    'Directory where Covertype data is stored.')
flags.DEFINE_string(
    'census_data',
    os.path.join(base_route, data_route, 'USCensus1990.data.txt'),
    'Directory where Census data is stored.')


def sample_data(data_type, num_contexts=None):
  """Sample data from given 'data_type'.

  Args:
    data_type: Dataset from which to sample.
    num_contexts: Number of contexts to sample.

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act).
    opt_rewards: Vector of expected optimal reward for each context.
    opt_actions: Vector of optimal action for each context.
    num_actions: Number of available actions.
    context_dim: Dimension of each context.
  """

  if data_type == 'linear':
    # Create linear dataset
    num_actions = 8
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                num_actions, sigma=noise_stds)
    opt_rewards, opt_actions = opt_linear
  elif data_type == 'sparse_linear':
    # Create sparse linear dataset
    num_actions = 7
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    num_nnz_dims = int(context_dim / 3.0)
    dataset, _, opt_sparse_linear = sample_sparse_linear_data(
        num_contexts, context_dim, num_actions, num_nnz_dims, sigma=noise_stds)
    opt_rewards, opt_actions = opt_sparse_linear
  elif data_type == 'mushroom':
    # Create mushroom dataset
    num_actions = 2
    context_dim = 117
    file_name = FLAGS.mushroom_data
    dataset, opt_mushroom = sample_mushroom_data(file_name, num_contexts)
    opt_rewards, opt_actions = opt_mushroom
  elif data_type == 'financial':
    num_actions = 8
    context_dim = 21
    num_contexts = min(3713, num_contexts)
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    file_name = FLAGS.financial_data
    dataset, opt_financial = sample_stock_data(file_name, context_dim,
                                               num_actions, num_contexts,
                                               noise_stds, shuffle_rows=True)
    opt_rewards, opt_actions = opt_financial
  elif data_type == 'jester':
    num_actions = 8
    context_dim = 32
    num_contexts = min(19181, num_contexts)
    file_name = FLAGS.jester_data
    dataset, opt_jester = sample_jester_data(file_name, context_dim,
                                             num_actions, num_contexts,
                                             shuffle_rows=True,
                                             shuffle_cols=True)
    opt_rewards, opt_actions = opt_jester
  elif data_type == 'statlog':
    file_name = FLAGS.statlog_data
    num_actions = 7
    num_contexts = min(43500, num_contexts)
    sampled_vals = sample_statlog_data(file_name, num_contexts,
                                       shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'adult':
    file_name = FLAGS.adult_data
    num_actions = 14
    num_contexts = min(45222, num_contexts)
    sampled_vals = sample_adult_data(file_name, num_contexts,
                                     shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'covertype':
    file_name = FLAGS.covertype_data
    num_actions = 7
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_covertype_data(file_name, num_contexts,
                                         shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'census':
    file_name = FLAGS.census_data
    num_actions = 9
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_census_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
  elif data_type == 'wheel':
    delta = 0.95
    num_actions = 5
    context_dim = 2
    mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
    std_v = [0.05, 0.05, 0.05, 0.05, 0.05]
    mu_large = 50
    std_large = 0.01
    dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta,
                                                  mean_v, std_v,
                                                  mu_large, std_large)
    opt_rewards, opt_actions = opt_wheel

  return dataset, opt_rewards, opt_actions, num_actions, context_dim


def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed after {} seconds.'.format(
    name, time.time() - t_init))
  print('---------------------------------------------------')

  performance_pairs = []
  for j, a in enumerate(algos):
    performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
  performance_pairs = sorted(performance_pairs,
                             key=lambda elt: elt[1],
                             reverse=True)
  for i, (name, reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
  print('---------------------------------------------------')
  print('---------------------------------------------------')


def main(_):

  np.random.seed(FLAGS.seed)
  tf.set_random_seed(FLAGS.seed)

  dt = datetime.datetime.now()
  timestr = '{}-{}-{}-{}'.format(dt.month, dt.day, dt.hour, dt.minute)
  FLAGS.logdir = os.path.join(FLAGS.logdir, timestr)

  # Problem parameters
  num_contexts = FLAGS.num_context

  # Data type in {linear, sparse_linear, mushroom, financial, jester,
  #                 statlog, adult, covertype, census, wheel}
  data_type = FLAGS.bandit

  # Create dataset
  sampled_vals = sample_data(data_type, num_contexts)
  dataset, opt_rewards, opt_actions, num_actions, context_dim = sampled_vals

  layer_sizes = [int(i) for i in FLAGS.layers.split(',')]

  # Define hyperparameters and algorithms
  hparams = tf.contrib.training.HParams(num_actions=num_actions)

  hparams_rms = tf.contrib.training.HParams(
      num_actions=num_actions,
      context_dim=context_dim,
      init_scale=0.3,
      activation=tf.nn.relu,
      layer_sizes=layer_sizes,
      batch_size=512,
      activate_decay=True,
      initial_lr=0.1,
      max_grad_norm=5.0,
      show_training=False,
      freq_summary=2000,
      buffer_s=-1,
      initial_pulls=2,
      optimizer='RMS',
      reset_lr=True,
      lr_decay_rate=0.5,
      training_freq=50,
      training_epochs=100,
      p=0.95,
      q=20)

  hparams_dropout = tf.contrib.training.HParams(
      num_actions=num_actions,
      context_dim=context_dim,
      init_scale=0.3,
      activation=tf.nn.relu,
      layer_sizes=layer_sizes,
      batch_size=512,
      activate_decay=True,
      initial_lr=0.1,
      max_grad_norm=5.0,
      show_training=False,
      freq_summary=2000,
      buffer_s=-1,
      initial_pulls=2,
      optimizer='RMS',
      reset_lr=True,
      lr_decay_rate=0.5,
      training_freq=50,
      training_epochs=100,
      use_dropout=True,
      keep_prob=0.80)

  hparams_bbb = tf.contrib.training.HParams(
      num_actions=num_actions,
      context_dim=context_dim,
      init_scale=0.3,
      activation=tf.nn.relu,
      layer_sizes=layer_sizes,
      batch_size=512,
      activate_decay=True,
      initial_lr=0.1,
      max_grad_norm=5.0,
      show_training=True,
      freq_summary=2000,
      buffer_s=-1,
      initial_pulls=2,
      optimizer='RMS',
      use_sigma_exp_transform=True,
      cleared_times_trained=20,
      initial_training_steps=2000,
      noise_sigma=0.1,
      reset_lr=False,
      training_freq=50,
      training_epochs=100)

  hparams_gp = tf.contrib.training.HParams(
      num_actions=num_actions,
      num_outputs=num_actions,
      context_dim=context_dim,
      reset_lr=False,
      learn_embeddings=True,
      max_num_points=1000,
      show_training=False,
      freq_summary=2000,
      batch_size=512,
      keep_fixed_after_max_obs=True,
      training_freq=50,
      initial_pulls=2,
      training_epochs=100,
      lr=0.01,
      buffer_s=-1,
      initial_lr=0.001,
      lr_decay_rate=0.0,
      optimizer='RMS',
      task_latent_dim=5,
      activate_decay=False)

  hparams_srld = tf.contrib.training.HParams(
    num_actions=num_actions,
    context_dim=context_dim,
    layer_sizes=layer_sizes,
    batch_size=512,
    activate_decay=False,
    initial_lr=0.1,
    lr=1e-7,
    lr_stein=1,
    show_training=True,
    freq_summary=2000,
    buffer_s=-1,
    initial_pulls=2,
    optimizer='Adam',
    use_sigma_exp_transform=True,
    cleared_times_trained=20,
    initial_training_steps=2000,
    noise_sigma=0.1,
    reset_lr=False,
    training_freq=50,
    training_epochs=100,
    n_particles=1,
    interp_batch_size=0,
    prior_variance=FLAGS.prior_variance)

  hparams_ld = tf.contrib.training.HParams(
    num_actions=num_actions,
    context_dim=context_dim,
    layer_sizes=layer_sizes,
    batch_size=512,
    activate_decay=False,
    initial_lr=0.1,
    lr=1e-7,
    lr_stein=0,
    show_training=True,
    freq_summary=2000,
    buffer_s=-1,
    initial_pulls=2,
    optimizer='Adam',
    use_sigma_exp_transform=True,
    cleared_times_trained=20,
    initial_training_steps=2000,
    noise_sigma=0.1,
    reset_lr=False,
    training_freq=50,
    training_epochs=100,
    n_particles=1,
    interp_batch_size=0,
    prior_variance=FLAGS.prior_variance)

  hparams_svgd = tf.contrib.training.HParams(
    num_actions=num_actions,
    context_dim=context_dim,
#   activation=tf.nn.relu,
    layer_sizes=layer_sizes,
    batch_size=512,
    activate_decay=False,
    initial_lr=0.1,
    lr=1e-6,
#   max_grad_norm=5.0,
    show_training=True,
    freq_summary=2000,
    buffer_s=-1,
    initial_pulls=2,
    optimizer='Adam',
    use_sigma_exp_transform=True,
    cleared_times_trained=20,
    initial_training_steps=2000,
    noise_sigma=0.1,
    reset_lr=False,
    training_freq=50,
    training_epochs=100,
    n_particles=20,
    interp_batch_size=0,
    prior_variance=FLAGS.prior_variance)

  algos = [
      UniformSampling('Uniform Sampling', hparams),
      PosteriorBNNSampling('srld', hparams_srld, 'SRLD'),
      PosteriorBNNSampling('ld', hparams_ld, 'SRLD'),
      PosteriorBNNSampling('SVGD', hparams_svgd, 'SVGD'),
  ]

  # Run contextual bandit problem
  t_init = time.time()
  results = run_contextual_bandit(
      context_dim, num_actions, dataset, algos, opt_rewards)
  _, h_rewards = results

  # Display results
  display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, data_type)

if __name__ == '__main__':
  app.run(main)
