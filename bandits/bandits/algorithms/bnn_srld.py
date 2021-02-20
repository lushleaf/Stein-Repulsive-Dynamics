import tensorflow as tf
import numpy as np
import zhusuan as zs

from bandits.algorithms.rf import init_bnn_weight
from bandits.algorithms.srld import *
from bandits.algorithms.utils import *

tfd = tf.contrib.distributions  # update to: tensorflow_probability.distributions


@zs.meta_bayesian_net(scope="bnn", reuse_variables=True)
def bnn_meta_model(x, layer_sizes, n_particles, n_supervised, w_prior_sd):
  """
  :param x: [batch_size, n_dims] (?)
  :param layer_sizes: list of length n_layers+1
  """
  assert int(x.shape[1]) == layer_sizes[0]

  bn = zs.BayesianNet()
  h = tf.tile(x[None, ...], [n_particles, 1, 1])

  w_scale = w_prior_sd * tf.ones([n_particles], dtype=tf.float32)

  for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    sd = tf.tile(w_scale[:, None, None], [1, n_in+1, n_out])
    if i != len(layer_sizes) - 2:
      sd /= np.sqrt(n_out).astype('f')
    w = bn.normal(
      "w" + str(i), tf.zeros([n_particles, n_in+1, n_out]), std=sd,
      group_ndims=2)
    h = tf.concat([h, tf.ones(tf.shape(h)[:-1])[..., None]], -1) @ w
    if i != len(layer_sizes) - 2:
      h = tf.nn.relu(h)

  y_mean = bn.deterministic("y_mean_sup", h[:, :n_supervised])
  y_std = bn.inverse_gamma(
    "y_std",
    alpha=tf.ones([1, layer_sizes[-1]]),
    beta=0.1 * tf.ones([1, layer_sizes[-1]]),
    n_samples=n_particles,
    group_ndims=1)
  y = bn.normal("y", y_mean, std=y_std)

  return bn

def build_bnn(x_ph, y_ph, weight_ph, n_train_ph, hps):

  inp, n_supervised = inplace_perturb(x_ph, hps.interp_batch_size, n_train_ph)
  layer_sizes = [x_ph.get_shape().as_list()[1]] + hps.layer_sizes + \
    [y_ph.get_shape().as_list()[1]]
  out_mask = weight_ph[None, :n_supervised, :]

  # ============== MODEL =======================
  weight_sd = np.sqrt(hps.prior_variance)
  meta_model = bnn_meta_model(
    inp, layer_sizes, hps.n_particles, n_supervised, weight_sd)

  w_names = ["w" + str(i) for i in range(len(layer_sizes) - 1)]
  def log_likelihood_fn(bn):
    log_pws = bn.cond_log_prob(w_names)
    log_py_xw = bn.cond_log_prob('y')
    log_py_xw = tf.reduce_sum(log_py_xw * out_mask, axis=-1)
    return tf.add_n(log_pws) + tf.reduce_mean(log_py_xw, 1) * n_train_ph

  meta_model.log_joint = log_likelihood_fn

  srld_var = dict()
  srld_latent = dict()

  if hps.use_sigma_exp_transform:
    sigma_transform = tfd.bijectors.Exp()
  else:
    sigma_transform = tfd.bijectors.Softplus()

  std_raw = tf.get_variable(
    'std_raw', shape=[hps.n_particles, 1, layer_sizes[-1]],
    initializer=tf.zeros_initializer())
  srld_var['y_std'] = std_raw
  y_std_sym = sigma_transform.forward(
    std_raw + sigma_transform.inverse(hps.noise_sigma))

  for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    w_init = init_bnn_weight(hps.n_particles, n_in, n_out) * weight_sd
    buf = tf.get_variable('buf_'+str(i), initializer=w_init)
    srld_latent['w'+str(i)] = srld_var['w'+str(i)] = buf

  # TODO: y_std should be sampled over time
  observed_bn_ = {'y': y_ph[:n_supervised], 'y_std':y_std_sym}

  sgmcmc = SGSRLD(lr_ld=hps.lr, lr_stein=hps.lr_stein, num_stein_particles=20, particle_update_freq=100)
  sample_op, sgmcmc_info = sgmcmc.sample(meta_model, observed=observed_bn_, latent=srld_latent)

  stein_particles = sgmcmc_info.stein_particles
  slice_index = tf.random.uniform([1], maxval=20, dtype=tf.dtypes.int32)
  srld_latent = dict(zip(list(stein_particles.keys()), [stein_particles[key][slice_index[0]] for key in list(stein_particles.keys())]))

  observed_bn_.update(srld_latent)
  var_bn = meta_model.observe(**observed_bn_)

  global_step = tf.get_variable(
    'global_step', initializer=0, trainable=False)

  log_py_xw = tf.reduce_sum(var_bn.cond_log_prob("y") * out_mask, axis=-1)
  log_likelihood = tf.reduce_mean(zs.log_mean_exp(log_py_xw, 0))
  y_pred = tf.reduce_mean(var_bn['y_mean_sup'], axis=0)
  rmse = tf.sqrt(tf.reduce_mean(
    (y_pred - y_ph[:n_supervised]) ** 2 * weight_ph[:n_supervised]))

  logs = {
    'rmse': rmse,
    'log_likelihood': log_likelihood,
    'mean_std': tf.reduce_mean(y_std_sym),
    'std_first': y_std_sym[0,0,0],
    'std_last': y_std_sym[0,0,-1]
  }

  return sample_op, var_bn['y_mean_sup'], locals()


def inplace_perturb(x, extra_batch_size, n_train, ptb_scale=0.1):
  if extra_batch_size == 0:
    return x, None

  x_dim = int(x.shape[1])
  ptb_scale = tf.to_float(ptb_scale / tf.sqrt(n_train * x_dim))
  to_perturb = x[-extra_batch_size:, :]
  x = x[:-extra_batch_size, :]
  x_perturbed = to_perturb + tf.random_normal(tf.shape(to_perturb), stddev=ptb_scale)
  model_inp = tf.stop_gradient(tf.concat([x, x_perturbed], axis=0))
  return model_inp, tf.shape(x)[0]

