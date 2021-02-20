import numpy as np
import tensorflow as tf

# SRLD using Global Sample for Repulsive

class SRLD:
    def __init__(self, num_p, seed=1234):
        """
        :param num_p: dimension of target distribution
        :param seed: seed used for reproduction
        """
        self.num_p = num_p
        self.seed = seed
        # isinitialized is used to record whether we have initialized
        self.isinitialized = 0

        # Distribution for initialization
    def weight_variable(self, shape, seed):
        initial = tf.truncated_normal(shape, mean=2.0, stddev=1.0, dtype=tf.float32, seed=seed)
        return tf.Variable(initial)

    def Init(self, num_N=50):
        """
        :param num_N: the number of particles used in SVGD operator
        :return: Initialization of particle
        """

        np.random.seed(self.seed)

        if not self.isinitialized:
            print('First time init')
            self.isinitialized = 1
            self.num_N = num_N
            self.theta = self.weight_variable([self.num_p, 1], self.seed)
            # <- close --------- time ----------- far -> #
            self.Record_N = np.zeros([self.num_p, self.num_N], dtype=np.float32)
            self.N = tf.Variable(self.Record_N, trainable=False)


            # Record the initialization
            with tf.Session() as sess:
                tf.set_random_seed(self.seed)
                tf.global_variables_initializer().run()
                self.Record_init_theta = sess.run(self.theta)

            print('init = ', self.Record_init_theta)

        # Re-initialize, keep the previous init value
        else:
            print('Re-init')
            self.num_N = num_N
            # <-close -------------- far-> #
            self.Record_N = np.zeros([self.num_p, self.num_N], dtype=np.float32)

            with tf.Session() as sess:
                tf.set_random_seed(self.seed)
                tf.global_variables_initializer().run()
                sess.run([tf.assign(self.theta, self.Record_init_theta), tf.assign(self.N, self.Record_N)])

                print('init = ', sess.run(self.theta))
    def SetSampler(self, Distribution, lr1 = 5e-3, lr2 = 5e-2, Decay_factor = 1.0, h_factor = 1.0):
        """
        :param lr1: learning rate of Langevin Dynamics
        :param lr2: learning rate of Stein Repulsive Gradient
        :param Decay_factor: (exponential) decay factor of the weight of particle distribution
        :param h_factor: used to scale the h calculated by median trick
        :param Distribution: Distribution to Sample, predefined
        :return: the graph used for sampling method
        """
        self.Distribution = Distribution
        self.lr1 = lr1
        self.lr2 = lr2
        self.Decay_factor = Decay_factor
        self.h_factor = h_factor
        self.sqrtl1 = np.sqrt(2.0*self.lr1)
        # Normalized weight for particle distribution in Stein Repulsive Potential
        # <-close ---------time----------- far-> #
        self.Rep_W = np.array([np.logspace(0.0, self.num_N - 1, self.num_N, base=self.Decay_factor, dtype=np.float32)])
        self.Rep_W = self.Rep_W / np.sum(self.Rep_W)
        self.Rep_W = tf.constant(self.Rep_W)

        # ------------------------- Target Distribution ----------------------------- #
        # ln prob distribution
        # change both of them!
        if Distribution == 'Normal':
            # Simple Normal Distribution
            self.lnp = -tf.reduce_sum(tf.square(self.theta))
            self.lnp_N = -tf.reduce_sum(tf.square(self.N), 0)

        elif Distribution == 'Mixture Normal':

            # Mixture of Normal Distribution

            self.com1 = tf.exp(
                -tf.reduce_sum(tf.square(self.theta - tf.constant(1.0 * np.ones([self.num_p, 1], dtype=np.float32))))*0.5)
            self.com2 = tf.exp(
                -tf.reduce_sum(tf.square(self.theta - tf.constant(-1.0 * np.ones([self.num_p, 1], dtype=np.float32))))*0.5)


            self.comN1 = tf.exp(
                -tf.reduce_sum(tf.square(self.N - tf.constant(1.0 * np.ones([self.num_p, self.num_N], dtype=np.float32))),0)*0.5)
            self.comN2 = tf.exp(
                -tf.reduce_sum(tf.square(self.N - tf.constant(-1.0 * np.ones([self.num_p, self.num_N], dtype=np.float32))),0)*0.5)

            self.lnp = tf.log(tf.multiply(1.0 / 2.0, self.com1) + tf.multiply(1.0 / 2.0, self.com2))
            self.lnp_N = tf.log(tf.multiply(1.0 / 2.0, self.comN1) + tf.multiply(1.0 / 2.0, self.comN2))

        elif Distribution == 'Rosenbrock':

            #2d Rosenbrock-like Distribution
            # parameter of Rosenbrock distribution
            #self.Rosenbrock_a = 1.0 / 20.0
            #self.Rosenbrock_b = 100 / 20.0
            """
            self.Rosenbrock_a = 1.0 / 0.1
            self.Rosenbrock_b = 100.0 / 0.1
            self.Rosenbrock_mu = 1.0
            self.lnp = -tf.square(tf.multiply(self.Rosenbrock_a, self.theta[0] - self.Rosenbrock_mu))\
                       -tf.square(tf.multiply(self.Rosenbrock_b, self.theta[1] - tf.square(self.theta[0])))\
                       #-tf.reduce_sum(tf.square(self.theta))
            self.lnp_N = -tf.square(tf.multiply(self.Rosenbrock_a, self.N[0, :] - self.Rosenbrock_mu)) \
                         -tf.square(tf.multiply(self.Rosenbrock_b, self.N[1, :] - tf.square(self.N[0, :])))\
                         #-tf.reduce_sum(tf.square(self.N))
            """

            """
            p(theta) ~ e^(-U(theta))
            U(theta) = theta_4^4/a1 + (a2(theta_2+b_1)-a_3*theta_1^2)^2/a_4
            """
            self.Rosenbrock_a1 = 10.0
            self.Rosenbrock_a2 = 4.0
            self.Rosenbrock_b1 = 1.2
            self.Rosenbrock_a3 = 1.0
            self.Rosenbrock_a4 = 2.0

            self.lnp = - self.theta[0] ** 4 / self.Rosenbrock_a1 - \
                       (self.Rosenbrock_a2 * (self.theta[1] + self.Rosenbrock_b1) - self.Rosenbrock_a3 * self.theta[
                       0] ** 2) ** 2 / self.Rosenbrock_a4

            self.lnp_N = - self.N[0, :] ** 4 / self.Rosenbrock_a1 - \
                       (self.Rosenbrock_a2 * (self.N[1, :] + self.Rosenbrock_b1) - self.Rosenbrock_a3 * self.N[
                       0, :] ** 2) ** 2 / self.Rosenbrock_a4


        else:
            raise ValueError('No predefined Distribution', self.Distribution)


        # ------------------------- Langevin Gradient ------------------------------- #
        self.gradP = tf.gradients(self.lnp, self.theta)[0]
        # Pretrain step
        # Draft term
        self.pretrain_step1 = tf.assign(self.theta, tf.add(self.theta, tf.multiply(self.gradP, self.lr1)))
        # Diffusion term
        self.pretrain_noise = tf.random_normal([self.num_p, 1], mean=0.0, stddev=self.sqrtl1, dtype=tf.float32, seed=self.seed)
        self.pretrain_step2 = tf.assign(self.theta, tf.add(self.theta, self.pretrain_noise))
        # -------------------------- Stein Repulsive Gradient ----------------------- #
        """
        h_dist: num_p * num_N, squared distance between current particle and previous particles
        h_median: num_N * 1, median of h_dist
        self.h: bandwidth calculated based on median trick

        Grad_N: p*num_N, gradient of previous particles
        temK1: 1*num_N, matrix of Kernel(x^l, x), x^l is previous particle
        """
        # Median Trick
        self.h_dist = tf.reduce_sum(tf.square(self.N - self.theta), 0, keepdims=True)
        self.h_median = tf.nn.top_k(self.h_dist[0], k=self.num_N // 2 + 1)[0][-1]
        self.h = tf.divide(self.h_median, self.h_factor*np.log(self.num_N))

        # Gradient
        self.Grad_N = tf.gradients(self.lnp_N, self.N)[0]
        self.temK1 = tf.exp(tf.divide(-self.h_dist, self.h))
        # Refining Potential
        self.K1 = tf.matmul(self.Grad_N, tf.transpose(tf.multiply(self.temK1, self.Rep_W)))
        # Repulsive Potential
        self.K2 = tf.multiply(tf.divide(-2.0, self.h), tf.matmul(self.N - self.theta, tf.transpose(tf.multiply(self.temK1, self.Rep_W))))
        self.gradK = self.K1 + self.K2

        # --------------------------- Train Operator ------------------------------- #
        self.gradTotal = tf.add(tf.multiply(self.gradP, self.lr1), tf.multiply(self.gradK, self.lr2))

        self.train_step1 = tf.assign(self.theta, tf.add(self.theta, self.gradTotal))
        self.train_noise = tf.random_normal([self.num_p, 1], mean=0.0, stddev=self.sqrtl1, dtype=tf.float32, seed=self.seed)
        self.train_step2 = tf.assign(self.theta, tf.add(self.theta, self.train_noise))

        # ---------------------- Particle Distribution Update Operator ------------- #
        self.update_N_value = tf.placeholder("float32", [self.num_p, self.num_N], name="holdN")
        self.update_N = tf.assign(self.N, self.update_N_value)

    def RunSampler(self, MaxLoop = 200, inter = 100):
        """
        :param MaxLoop: number of samples
        :param MaxLoop: samples in between
        :return:
        """
        display_inter = 1000

        np.random.seed(self.seed)

        self.inter = inter
        self.MaxLoop = MaxLoop
        self.theta_path = np.zeros([self.num_p, self.MaxLoop])
        self.GradV_path = np.zeros([self.num_p, self.MaxLoop])
        self.GradK_path = np.zeros([self.num_p, self.MaxLoop])


        with tf.Session() as sess:

            tf.set_random_seed(self.seed)
            tf.global_variables_initializer().run()

            for iti in range(self.MaxLoop*self.inter):
                if self.lr2 !=0:
                    if iti <= self.inter:
                        itirun = sess.run([self.pretrain_step1, self.pretrain_step2, self.theta])
                        if iti % self.inter == 0:
                            if np.isnan(itirun[2]).any():
                                raise ValueError('Numerical Error: Nan')
                            self.theta_path[:, [iti//self.inter]] = itirun[2]

                            num_effect_sample = iti // self.inter + 1
                            num_rep = self.num_N // num_effect_sample + 1
                            self.Record_N[:, 0:self.num_N] = np.repeat(self.theta_path[:, 0:num_effect_sample], num_rep,
                                                                       axis=1)[:, 0:self.num_N]
                            sess.run(self.update_N, feed_dict={self.update_N_value: self.Record_N})
                            temitirun = sess.run([self.gradP])
                            self.GradV_path[:, [iti // self.inter]] = temitirun[0]

                    else:
                        itirun = sess.run([self.train_step1, self.train_step2, self.theta])
                        if iti % self.inter == 0:
                            if np.isnan(itirun[2]).any():
                                raise ValueError('Numerical Error: Nan')
                            self.theta_path[:, [iti // self.inter]] = itirun[2]

                            temitirun = sess.run([self.gradP, self.gradK])
                            self.GradV_path[:, [iti // self.inter]] = temitirun[0]
                            self.GradK_path[:, [iti // self.inter]] = temitirun[1]

                            num_effect_sample = iti // self.inter + 1
                            if num_effect_sample < self.num_N:
                                num_rep = self.num_N // num_effect_sample + 1
                                self.Record_N[:, 0:self.num_N] = np.repeat(self.theta_path[:, 0:num_effect_sample], num_rep,
                                                                           axis=1)[:, 0:self.num_N]
                            else:
                                self.Record_N[:, 0:self.num_N] = self.theta_path[:, num_effect_sample - self.num_N:num_effect_sample]

                            sess.run(self.update_N, feed_dict={self.update_N_value: self.Record_N})
                else:
                    itirun = sess.run([self.pretrain_step1, self.pretrain_step2, self.theta])
                    if iti % self.inter == 0:
                        if np.isnan(itirun[2]).any():
                            raise ValueError('Numerical Error: Nan')

                        self.theta_path[:, [iti // self.inter]] = itirun[2]
                        temitirun = sess.run([self.gradP])
                        self.GradV_path[:, [iti // self.inter]] = temitirun[0]

                if iti % display_inter == 0:
                    print('Pretrain: Num of Iteration = ', iti)

        return self.theta_path, self.GradV_path, self.GradK_path


