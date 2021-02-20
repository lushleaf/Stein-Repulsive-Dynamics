from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import zhusuan as zs
import tensorflow as tf

def rbf_kernel(theta_x, theta_y, bandwidth='median'):
    """
    :param theta: tensor of shape [n_particles, n_params]
    :return: tensor of shape [n_particles, n_particles]
    """
    n_x = tf.shape(theta_x)[0]
    pairwise_dists = tf.reduce_sum(
        (tf.expand_dims(theta_x, 1) - tf.expand_dims(theta_y, 0)) ** 2,
        axis=-1)
    if bandwidth == 'median':
        bandwidth = tf.contrib.distributions.percentile(
            tf.squeeze(pairwise_dists), q=50.)
        bandwidth = 0.5 * bandwidth / tf.log(tf.cast(n_x, tf.float32) + 1)
        bandwidth = tf.maximum(tf.stop_gradient(bandwidth), 1e-5)
    Kxy = tf.exp(-pairwise_dists / bandwidth / 2)
    return Kxy, -tf.gradients(Kxy, theta_x)[0]

def _squeeze(tensors, n_particles):
    return tf.concat(
        [tf.reshape(t, [n_particles, -1]) for t in tensors], axis=1)


def _unsqueeze(squeezed, original_tensors):
    ret = []
    offset = 0
    for t in original_tensors:
        size = tf.reduce_prod(tf.shape(t)[1:])
        buf = squeezed[:, offset: offset+size]
        offset += size
        ret.append(tf.reshape(buf, tf.shape(t)))
    return ret

class SGSRLD(zs.SGMCMC):
    def __init__(self, lr_ld, lr_stein, num_stein_particles, particle_update_freq):
        self.lr_ld = tf.convert_to_tensor(
            lr_ld, tf.float32, name="lr_ld")
        self.lr_stein = tf.convert_to_tensor(
            lr_stein, tf.float32, name="lr_stein")
        self.num_stein_particles = num_stein_particles
        self.particle_update_freq = particle_update_freq
        super(SGSRLD, self).__init__()

    def _define_variables(self, qs):
        self.stein_particles = []
        for i in range(len(qs)):
            self.stein_particles.append(tf.Variable(tf.zeros([self.num_stein_particles] + qs[i].shape.dims), trainable=False))

    def _update(self, qs, grad_func):
        update_q_op, infos = tf.cond(tf.less(self.t, self.num_stein_particles * self.particle_update_freq),
                     lambda: list(zip(*[self._update_single_ld(q, grad, stein_particle)
                     for q, grad, stein_particle in zip(qs, grad_func(qs), self.stein_particles)])),
                     lambda: self._update_srld(qs, grad_func))
        with tf.control_dependencies(update_q_op):
            update_par_op = tf.cond(tf.equal(tf.mod(self.t, self.particle_update_freq), 0),
                                    lambda:self._update_particles(qs),
                                    lambda:self._keep_particles())
        update_ops = [update_q_op, update_par_op]
        return update_ops, infos

    def _update_particles(self, qs):
        update_par_op = []
        for i in range(len(qs)):
            update_par_op.append(self.stein_particles[i].assign(tf.concat([self.stein_particles[i][1:], tf.expand_dims(qs[i], 0)], 0)))
        return update_par_op        
        
    def _keep_particles(self):
        update_par_op = []
        for i in range(len(self.stein_particles)):
            update_par_op.append(self.stein_particles[i].assign(self.stein_particles[i]))
        return update_par_op

    def _update_single_ld(self, q, grad, stein_particle):
        new_q = q + 0.5 * self.lr_ld * grad + tf.random_normal(
            tf.shape(q), stddev=tf.sqrt(self.lr_ld))
        update_q = q.assign(new_q)
        info = {"q": new_q, 'stein_particles':stein_particle}
        return update_q, info

    def _update_srld(self, qs, grad_func):
        n_particles = int(qs[0].shape[0])
        grads = grad_func(qs)
        particle_stein_grads = []
        for i in range(n_particles):
            qi_squeezed = _squeeze([q[i] for q in qs], 1)
            stein_particle_i = [stein_particle_i[:, i] for stein_particle_i in self.stein_particles]
            stein_particle_i_squeezed = _squeeze(stein_particle_i, self.num_stein_particles)

            Kxy, dxkxy = rbf_kernel(stein_particle_i_squeezed, tf.stop_gradient(qi_squeezed))

            num_eval = (self.num_stein_particles - 1) // n_particles + 1
            particle_grads_collection = []
            # Notice that we don't handle the case num_stein_particles % n_particles !=0 for simplicity, however in general we can handle it
            for j in range(num_eval):
                particle_grads_i_j = grad_func([variable[j*n_particles:(j+1)*n_particles] for variable in stein_particle_i])
                particle_grads_collection.append(particle_grads_i_j)

            particle_grads_i = []
            for j in range(len(qs)):
                particle_grads_i.append(tf.concat([grad[j] for grad in particle_grads_collection], axis=0))
                
            particle_grads_i_squeezed = _squeeze(particle_grads_i, self.num_stein_particles)
            stein_grads_i = (tf.matmul(tf.transpose(Kxy), particle_grads_i_squeezed) + tf.reduce_sum(dxkxy, 0, keep_dims=True)) / tf.to_float(self.num_stein_particles)
            unsqueezed_stein_grads = _unsqueeze(stein_grads_i, [tf.expand_dims(q[i], 0) for q in qs])
            particle_stein_grads.append(unsqueezed_stein_grads)
        
        stein_grads = []
        for i in range(len(qs)):
            stein_grads.append(tf.concat([particle_stein_grad[i] for particle_stein_grad in particle_stein_grads], axis=0))

        return list(zip(*[self._update_single_srld(q, grad, stein_particle, stein_grad)
                     for q, grad, stein_particle, stein_grad in zip(qs, grads, self.stein_particles, stein_grads)]))

    def _update_single_srld(self, q, grad, stein_particle, stein_gradient):
        new_q = q + 0.5 * self.lr_ld * (grad + self.lr_stein * stein_gradient) + tf.random_normal(
            tf.shape(q), stddev=tf.sqrt(self.lr_ld))
        update_q = q.assign(new_q)
        info = {"q": new_q, 'stein_particles':stein_particle}
        return update_q, info

