import tensorflow as tf
import numpy as np
import scipy.stats
import warnings
import time


# SRLD using Global Sample for Repulsive
# Sampling 2-layer Bayesian Neural Network

class SRLD_BNN:
    def __init__(self, seed=1234):
        """
        :param seed: seed used for reproduction
        """
        self.seed = seed
        # isinitialized is used to record whether we have initialized
        self.isinitialized = 0
        # isburnin is used to record whether we have burnin
        self.isburnin = 0

    def weight_variable(self, shape, seed, dist='TrunNormal'):
        if dist == 'TrunNormal':
            initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32, seed=seed)
            return tf.Variable(initial)
        elif dist == 'Xavier':
            initial = tf.random_uniform(shape, minval=-np.sqrt(6.0 / (shape[0] + shape[1])),
                                        maxval=np.sqrt(6.0 / (shape[0] + shape[1])), dtype=tf.float32, seed = seed)
            return tf.Variable(initial)
        else:
            raise ValueError('Initialization method has not been defined!')


    def Init(self,
             num_p,
             num_H = 50,
             num_N = 20):

        """
        -- Initialize network structure --
        :param num_p: dimension of input variable
        :param num_H: num of neurons
        :param num_N: num of particles used in SVGD operator
        :return: Initialize the neural network
        """

        np.random.seed(self.seed)

        if not self.isinitialized:
            print('First time init')
            self.isinitialized = 1

            self.num_p = num_p
            self.num_H = num_H
            self.num_N = num_N

            # Def/Init NN
            self.x = tf.placeholder("float32", [None, self.num_p])
            self.W1 = self.weight_variable([self.num_p, self.num_H], self.seed, dist='Xavier')
            self.H1 = tf.tanh(tf.matmul(self.x, self.W1))
            self.W2 = self.weight_variable([self.num_H, 1], self.seed, dist='Xavier')
            self.y = tf.matmul(self.H1, self.W2)
            self.yt = tf.placeholder("float32", [None, 1])

            # Def/Init Repulsive NN
            # x_N in R^(num_N, num_data, num_p)
            # y_N in R^(num_N, num_data, 1)

            self.x_N = tf.tile(tf.expand_dims(self.x, axis=0), multiples=[self.num_N, 1, 1])

            # <- close --------- time ----------- far -> #
            self.Record_W1_N = np.zeros([self.num_N, self.num_p, self.num_H], dtype=np.float32)
            self.W1_N = tf.Variable(self.Record_W1_N, trainable=False)
            self.H1_N = tf.tanh(tf.matmul(self.x_N, self.W1_N))

            # <- close --------- time ----------- far -> #
            self.Record_W2_N = np.zeros([self.num_N, self.num_H, 1], dtype=np.float32)
            self.W2_N = tf.Variable(self.Record_W2_N, trainable=False)
            self.y_N = tf.matmul(self.H1_N, self.W2_N)
            self.yt_N = tf.tile(tf.expand_dims(self.yt, axis=0), multiples=[self.num_N, 1, 1])

            # <- close --------- time ----------- far-> #
            self.Record_noise_sigma = np.zeros([1, num_N], dtype=np.float32)

            # Record the initialization
            with tf.Session() as sess:
                tf.set_random_seed(self.seed)
                tf.global_variables_initializer().run()

                itirun = sess.run([self.W1, self.W2])
                self.Record_init_W1 = itirun[0]
                self.Record_init_W2 = itirun[1]

            # Re-initialize, keep the previous init value
        else:
            print('Re-init')
            # <-close -------------- far-> #
            self.Record_W1_N = np.zeros([self.num_N, self.num_p, self.num_H], dtype=np.float32)
            self.Record_W2_N = np.zeros([self.num_N, self.num_H, 1], dtype=np.float32)
            self.Record_noise_sigma = np.zeros([1, num_N], dtype=np.float32)

            with tf.Session() as sess:
                tf.set_random_seed(self.seed)
                tf.global_variables_initializer().run()
                sess.run([tf.assign(self.W1, self.Record_init_W1),
                          tf.assign(self.W2, self.Record_init_W2),
                          tf.assign(self.W1_N, self.Record_W1_N),
                          tf.assign(self.W2_N, self.Record_W2_N)])


    def SetSampler(self,
                   num_train,
                   method = 'GD',
                   num_subsam = 100,
                   prior_sigma = 1.0,
                   IG_prior_a = 1.0,
                   IG_prior_b = 0.1,
                   h_factor=1.0,
                   Decay_factor=1.0,
                   isnormalize=0):

        """
        :param num_train: num of training samples
        :param method: optimization method --
                       'GD' = gradient descent;
                       'Momentum' = Momentum;
                       'AdaGrad' = AdaGrad;
                       'Adam' = Adam;
                       'RMSProp' = RMSProp;
        :param num_subsam: size of mini-batch
        :param prior_sigma: Varaince of the Gaussian Prior for weight
        :param IG_prior_a: Shape parameter of the IG prior for variance of data noise
        :param IG_prior_b: Scale parameter of the IG prior for variance of data noise
        :param h_factor: Magnitude of bandwidth of kernel calculated by median trick
        :param Decay_factor: Exponential Decay rate of the weight of past particles in SVGD Operator
        :return: Set the graph for sampler
        """
        self.num_train = num_train
        self.num_subsam = num_subsam
        self.method = method
        self.isnormalize = isnormalize

        """
        -- Prior -- :
        Independent prior on each coordinate of weight
        weight ~ N(0, prior_sigma); prior_sigma is the VARIANCE (not std) of the prior

        Inverse Gamma prior on the Variance of observation noise
        noise_sigma ~ IG(IG_prior_a, IG_prior_b)

        noise_sigma is initialized with 1.0 and sampled via Gibbs step
        """
        self.prior_sigma = prior_sigma
        self.IG_prior_a = IG_prior_a
        self.IG_prior_b = IG_prior_b

        self.noise_sigma = tf.placeholder("float32", shape=())
        self.noise_sigma_value = 1.0  # value of noise_sigma_value


        """
        -- Learning Rate -- :
        We scale the used lr by base_lr/num_train
        """

        self.lr1_hold = tf.placeholder("float32", shape=())
        self.lr2_hold = tf.placeholder("float32", shape=())
        self.sqrtl1_hold = tf.placeholder("float32", shape=())

        """
        -- SVGD Operator -- :
        h_factor scales the magnitude of the median trick used in choosing the bandwidth of kernel of SVGD operator
        Decay_factor is the (rate of exponential decay of the particle weight):

        phi[past distribution](w)
        = \sum_{i=1}^N { phi[w_{t-i}](w) * exp(-Decay_factor * i)} / \sum_{i=1}^N {exp(-Decay_factor * i)}

        Decay_factor is defaultly set to be 1.0 for non-decay scheme.
        """
        self.h_factor = h_factor
        self.Decay_factor = Decay_factor

        # Define the Prior/Posterior
        self.ln_prior = tf.reduce_sum(-tf.square(self.W1) / (2.0 * self.prior_sigma)) + \
                        tf.reduce_sum(-tf.square(self.W2) / (2.0 * self.prior_sigma))
        self.loss = -tf.reduce_sum(tf.square(self.yt - self.y))

        if self.isnormalize:
            self.std = tf.placeholder("float32", shape=())
            self.mean = tf.placeholder("float32", shape=())
            self.loss_unnorm = -tf.reduce_sum(tf.square(self.yt - (self.y*self.std) - self.mean))

        self.ln_likelihood = self.num_train / self.num_subsam * self.loss / (2.0 * self.noise_sigma)
        self.ln_post = self.ln_likelihood + self.ln_prior

        # Define the Prior/Posterir of the N-particel SVGD operator particels
        self.ln_prior_N = tf.reduce_sum(-tf.square(self.W1_N) / (2.0 * self.prior_sigma)) + \
                          tf.reduce_sum(-tf.square(self.W2_N) / (2.0 * self.prior_sigma), axis=[1, 2])
        self.loss_N = -tf.reduce_sum(tf.square(self.yt_N - self.y_N), axis=[1, 2])
        self.ln_likelihood_N = self.num_train / self.num_subsam * self.loss_N / (2.0 * self.noise_sigma)
        self.ln_post_N = self.ln_likelihood_N + self.ln_prior_N


        # Train Operator
        # ------------------------- Langevin Gradient ------------------------------- #
        self.gradP = tf.gradients(self.ln_post, [self.W1, self.W2])

        # gradP[0] is the gradient of W1,  gradP[1] is the gradient of W2, gradP[2] is the gradient of B2

        # Pretrain step: no SVGD operator is involved
        # Drafting term
        self.pretrain_step1 = [tf.assign(self.W1, tf.add(self.W1, tf.multiply(self.gradP[0], self.lr1_hold))),
                               tf.assign(self.W2, tf.add(self.W2, tf.multiply(self.gradP[1], self.lr1_hold)))]
        # Diffusion term
        self.pretrain_noise = [tf.random_normal([self.num_p, self.num_H],
                                                mean=0.0, stddev=self.sqrtl1_hold, dtype=tf.float32, seed=self.seed),
                               tf.random_normal([self.num_H, 1],
                                                mean=0.0, stddev=self.sqrtl1_hold, dtype=tf.float32, seed=self.seed)]
        self.pretrain_step2 = [tf.assign(self.W1, tf.add(self.W1, self.pretrain_noise[0])),
                               tf.assign(self.W2, tf.add(self.W2, self.pretrain_noise[1]))]

        # -------------------------- Stein Repulsive Gradient ----------------------- #
        """
        h_dist: 1 * num_N, squared distance between current particle and previous particles
        h_median: in R^1, median of h_dist
        self.h: bandwidth calculated based on median trick; Magnitude can be tuned by h_factor

        Grad_N: num_p*num_N, gradient of previous particles
        temK1: 1*num_N, matrix of Kernel(x^l, x), x^l is previous particle
        """

        # Set Weight
        # Normalized weight for particle distribution in Stein Repulsive Potential
        # <-close ---------time----------- far-> #
        self.Rep_W = np.array([np.logspace(0.0, self.num_N - 1, self.num_N, base=self.Decay_factor, dtype=np.float32)])
        self.Rep_W = self.Rep_W / np.sum(self.Rep_W)
        self.Rep_W = tf.constant(self.Rep_W)

        # Median Trick
        self.W1_expend = tf.tile(tf.expand_dims(self.W1, axis=0), multiples=[self.num_N, 1, 1])
        self.W2_expend = tf.tile(tf.expand_dims(self.W2, axis=0), multiples=[self.num_N, 1, 1])
        self.W1_h_dist = tf.reduce_sum(tf.square(self.W1_N - self.W1_expend), axis=[1, 2])
        self.W2_h_dist = tf.reduce_sum(tf.square(self.W2_N - self.W2_expend), axis=[1, 2])

        self.W1_h_median = tf.nn.top_k(self.W1_h_dist, k=self.num_N // 2 + 1)[0][-1]
        self.W2_h_median = tf.nn.top_k(self.W2_h_dist, k=self.num_N // 2 + 1)[0][-1]

        self.W1_h = tf.divide(self.W1_h_median, self.h_factor * np.log(self.num_N))
        self.W2_h = tf.divide(self.W2_h_median, self.h_factor * np.log(self.num_N))

        # Gradient
        if self.method == 'GD':
            Opt_N = tf.train.GradientDescentOptimizer(1.0)
        elif self.method == 'Momentum':
            Opt_N = tf.train.MomentumOptimizer(1.0, 0.9)
        elif self.method == 'AdaGrad':
            Opt_N = tf.train.AdagradOptimizer(1.0)
        elif self.method == 'Adam':
            Opt_N = tf.train.AdamOptimizer(1.0)
        elif self.method == 'RMSProp':
            Opt_N = tf.train.RMSPropOptimizer(1.0)
        else:
            raise ValueError('Optimization method not defined!')

        self.Grad_N = Opt_N.compute_gradients(self.ln_post_N, var_list = [self.W1_N, self.W2_N])
        # self.Grad_N[i] is the information of the i-th variable
        # self.Grad_N[i][0] is the gradient of the i-th variable
        # self.Grad_N[i][1] is the value of the i-th variable

        # self.Grad_N = tf.gradients(self.ln_post_N, [self.W1_N, self.W2_N])
        self.W1_temK1 = tf.exp(tf.divide(-self.W1_h_dist, self.W1_h))
        self.W2_temK1 = tf.exp(tf.divide(-self.W2_h_dist, self.W2_h))

        # Refining Potential
        self.W1_Weight = tf.reshape(tf.multiply(self.W1_temK1, self.Rep_W), [self.num_N, 1, 1])
        self.W2_Weight = tf.reshape(tf.multiply(self.W2_temK1, self.Rep_W), [self.num_N, 1, 1])

        self.Weight_W1 = tf.tile(self.W1_Weight, [1, self.num_p, self.num_H])
        self.Weight_W2 = tf.tile(self.W2_Weight, [1, self.num_H, 1])
        # self.W1_K1 = tf.reduce_sum(tf.multiply(self.Grad_N[0], self.Weight_W1), axis=0)
        # self.W2_K1 = tf.reduce_sum(tf.multiply(self.Grad_N[1], self.Weight_W2), axis=0)

        self.W1_K1 = tf.reduce_sum(tf.multiply(self.Grad_N[0][0], self.Weight_W1), axis=0)
        self.W2_K1 = tf.reduce_sum(tf.multiply(self.Grad_N[1][0], self.Weight_W2), axis=0)

        # Repulsive Potential
        self.W1_K2 = tf.reduce_sum(tf.multiply(tf.divide(-2.0, self.W1_h),
                                               tf.multiply(self.W1_N - self.W1_expend, self.Weight_W1)), axis=0)
        self.W2_K2 = tf.reduce_sum(tf.multiply(tf.divide(-2.0, self.W2_h),
                                               tf.multiply(self.W2_N - self.W2_expend, self.Weight_W2)), axis=0)

        self.W1_gradK = self.W1_K1 + self.W1_K2
        self.W2_gradK = self.W2_K1 + self.W2_K2

        # --------------------------- Train Operator ------------------------------- #
        self.W1_gradTotal = tf.add(tf.multiply(self.gradP[0], self.lr1_hold), tf.multiply(self.W1_gradK, self.lr2_hold))
        self.W2_gradTotal = tf.add(tf.multiply(self.gradP[1], self.lr1_hold), tf.multiply(self.W2_gradK, self.lr2_hold))

        self.train_step1 = [tf.assign(self.W1, tf.add(self.W1, self.W1_gradTotal)),
                            tf.assign(self.W2, tf.add(self.W2, self.W2_gradTotal))]

        self.train_noise = [tf.random_normal([self.num_p, self.num_H],
                                             mean=0.0, stddev=self.sqrtl1_hold, dtype=tf.float32, seed=self.seed),
                            tf.random_normal([self.num_H, 1],
                                             mean=0.0, stddev=self.sqrtl1_hold, dtype=tf.float32, seed=self.seed)]
        self.train_step2 = [tf.assign(self.W1, tf.add(self.W1, self.train_noise[0])),
                            tf.assign(self.W2, tf.add(self.W2, self.train_noise[1]))]

        # ---------------------- Particle Distribution Update Operator ------------- #
        self.update_W1_value = tf.placeholder("float32", [self.num_N, self.num_p, self.num_H])
        self.update_W2_value = tf.placeholder("float32", [self.num_N, self.num_H, 1])
        self.update_W = [tf.assign(self.W1_N, self.update_W1_value),
                         tf.assign(self.W2_N, self.update_W2_value)]


    def RunBurnin(self,
                  x_train = [],
                  y_train = [],
                  base_burnin_lr1 = 0.01,
                  num_burnin = 100,
                  thin_inter = 200,
                  gibbs_inter = 100,
                  isSelect = 0):

        """
        :param x_train: training set
        :param y_train: training set
        :param base_burnin_lr1: base_lr for burnin process (using langevin dynamics); The used lr = base_lr/num_train
        :param num_burnin: num of burnin sample
        :param thin_inter: thinning factor (num of iteration to collect one sample)
        :param gibbs_inter: num of iteration to update sigma
        :return: last sample of burnin and N-particle
        """

        self.isburnin = 1
        self.isSelect = isSelect

        self.x_train = x_train
        self.y_train = y_train


        self.burnin_lr1 = base_burnin_lr1/self.num_train
        self.burnin_sqrtl1 = 1.0 * np.sqrt(2.0 * self.burnin_lr1)
        self.num_burnin = num_burnin
        self.thin_inter = thin_inter
        self.gibbs_inter = gibbs_inter


        if self.x_train == [] or self.y_train == []:
            raise ValueError('Training data is empty')

        if self.num_burnin <= self.num_N:
            raise ValueError('num Burnin is less than num of particles')

        if self.isnormalize:
            self.y_train_unnorm = self.y_train
            self.y_train_mean = np.mean(self.y_train)
            self.y_train_std = np.std(self.y_train)
            self.y_train = (self.y_train - self.y_train_mean)/self.y_train_std

        # ------------------------- Start Sampling ---------------------------- #
        np.random.seed(self.seed)

        with tf.Session() as sess:
            tf.set_random_seed(self.seed)
            tf.global_variables_initializer().run()
            for iti in range(self.num_burnin * self.thin_inter):
                n1 = (iti * self.num_subsam) % self.num_train
                if n1 + self.num_subsam <= self.num_train:
                    n2 = n2 = n1 + self.num_subsam
                    subsam_range = np.arange(n1, n2)
                else:
                    subsam_range1 = np.arange(n1, self.num_train)
                    subsam_range2 = np.arange(0, ((iti + 1) * self.num_subsam) % self.num_train)
                    subsam_range = np.append(subsam_range1, subsam_range2)


                # -------------- Step 1: Gibbs sampler of noise_sigma ---------------- #
                if iti % self.gibbs_inter == 0:
                    temitirun = sess.run(self.loss, feed_dict={self.x: self.x_train, self.yt: self.y_train})

                    # error check
                    if -temitirun / self.num_train >= 1e8 or np.isnan(temitirun):
                        print('MSE = ', -temitirun / self.num_train)
                        print('loss numeric error')
                        return

                    self.noise_sigma_value = \
                        self.Inverse_Gamma_Post(self.IG_prior_a, self.IG_prior_b, self.num_train, -temitirun,
                                                seed=self.seed)

                    if iti // self.thin_inter < self.num_N:
                        self.Record_noise_sigma[:, iti // self.thin_inter] = self.noise_sigma_value
                    else:
                        self.Record_noise_sigma[:, 1:self.num_N] = self.Record_noise_sigma[:, 0:self.num_N - 1]
                        self.Record_noise_sigma[:, 0] = self.noise_sigma_value

                # -------------------- Step 2: Update Weight ------------------------ #
                itirun = sess.run([self.pretrain_step1, self.pretrain_step2],
                                  feed_dict={self.noise_sigma: self.noise_sigma_value,
                                             self.x: self.x_train[subsam_range, :],
                                             self.yt: self.y_train[subsam_range, :],
                                             self.lr1_hold: self.burnin_lr1,
                                             self.sqrtl1_hold: self.burnin_sqrtl1})

                if iti % self.thin_inter == 0:
                    temitirun = sess.run([self.W1, self.W2])

                    # check numeric error
                    if np.isnan(temitirun[0]).any() or np.isnan(temitirun[1]).any() or not self.noise_sigma_value:
                        print('weight/sigma numeric error')
                        return

                        # raise ValueError('Numerical Error: Nan')

                    num_effect_sample = iti // self.thin_inter + 1
                    if num_effect_sample < self.num_N:
                        # Update Record_W1_N/Record_W2_N
                        num_effect_sample = iti // self.thin_inter + 1
                        self.Record_W1_N[[iti // self.thin_inter], :, :] = temitirun[0]
                        self.Record_W2_N[[iti // self.thin_inter], :, :] = temitirun[1]
                        num_rep = self.num_N // num_effect_sample + 1
                        tem_Record_W1_N = \
                            np.repeat(self.Record_W1_N[0:num_effect_sample, :, :], num_rep, axis=0)[
                            0:self.num_N, :, :]
                        tem_Record_W2_N = \
                            np.repeat(self.Record_W2_N[0:num_effect_sample, :, :], num_rep, axis=0)[
                            0:self.num_N, :, :]
                        sess.run(self.update_W, feed_dict={self.update_W1_value: tem_Record_W1_N,
                                                           self.update_W2_value: tem_Record_W2_N})
                        del tem_Record_W1_N
                        del tem_Record_W2_N
                    else:
                        self.Record_W1_N[1:self.num_N, :, :] = self.Record_W1_N[0:self.num_N - 1, :, :]
                        self.Record_W1_N[0, :, :] = temitirun[0]
                        self.Record_W2_N[1:self.num_N, :, :] = self.Record_W2_N[0:self.num_N - 1, :, :]
                        self.Record_W2_N[0, :, :] = temitirun[1]
                        sess.run(self.update_W, feed_dict={self.update_W1_value: self.Record_W1_N,
                                                           self.update_W2_value: self.Record_W2_N})

                        # Show Burn-in Process
                if iti % self.thin_inter == 0:
                    print('Burn-in')
                    print('iteration ', iti, '/', self.thin_inter * self.num_burnin)
                    print('noise_sigma =', self.noise_sigma_value)
                    print('---------------------------------------------')

            self.ctd_Record_W1_N = self.Record_W1_N
            self.ctd_Record_W2_N = self.Record_W2_N
            self.ctd_Record_noise_sigma = self.Record_noise_sigma

            if not self.isSelect:
                self.ctd_W1 = sess.run(self.W1)
                self.ctd_W2 = sess.run(self.W2)
                self.ctd_noise_sigma_value = self.noise_sigma_value
            else:
                fin_mse = 1e8
                fin_sel = -1
                for sel in range(self.num_N):
                    sess.run([tf.assign(self.W1, self.Record_W1_N[sel, :, :]),
                              tf.assign(self.W2, self.Record_W2_N[sel, :, :])])

                    tem_mse = sess.run(self.loss, feed_dict={self.x: self.x_train, self.yt: self.y_train})
                    tem_mse = -tem_mse/self.num_train
                    print ('tem_mse = ', tem_mse)
                    if tem_mse <= fin_mse:
                        print('new selection')
                        fin_sel = sel
                        fin_mse = tem_mse

                self.ctd_W1 = self.Record_W1_N[fin_sel, :, :]
                self.ctd_W2 = self.Record_W2_N[fin_sel, :, :]
                self.ctd_noise_sigma_value = self.Record_noise_sigma[:, fin_sel]



    def RunSampler(self,
                   x_val=[],
                   y_val=[],
                   x_test=[],
                   y_test=[],
                   base_lr1=0.01,
                   base_lr2=0.01,
                   num_sample = 400,
                   display_inter = 200,
                   Grad_inter=-1):

        """
        :param x_val: validation set, x; can be empty
        :param y_val: validation set, y can be empty
        :param x_test: test set, x; can be empty
        :param y_test: test set, y; can be empty
        :param num_sample: num of sample to collect (incluing burnin); must > num_burnin
        :param display_inter: num of iteration to calculate prediction/loss
        :param Grad_inter: num of iteration to estimate the gradients ratio of Langevin/SVGD; if <=0: not estimate
        :return: predictions and loss
        """

        if not self.isburnin:
            raise ValueError('Burn-in First!')

        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        self.num_sample = num_sample
        self.display_inter = display_inter
        self.Grad_inter = Grad_inter

        self.lr1 = base_lr1/self.num_train
        self.sqrtl1 = 1.0 * np.sqrt(2.0 * self.lr1)
        self.lr2 = base_lr2/self.num_train

        display_count = 0

        if self.num_sample <= self.num_burnin:
            warnings.warn('num_sample is less than the num of burnin!')

        # We generally require num_burnin is larger than the num of particles
        if self.lr2 != 0 and self.num_burnin < self.num_N:
            warnings.warn('num_burnin is less than the num of paticles!')

        # Display inter can not be too large (and no sample will be collected)
        if display_inter > num_sample * self.thin_inter:
            warnings.warn('display_inter is too large: No sample will be collected!')

        self.sigma_path = np.zeros(((self.num_sample - self.num_burnin) * self.thin_inter) // self.display_inter - 1 + 1)

        self.train_predict_path = \
            np.zeros([self.num_train, ((self.num_sample - self.num_burnin) * self.thin_inter) // self.display_inter - 1 + 1])
        self.train_loss_path = \
            np.zeros([1, ((self.num_sample - self.num_burnin) * self.thin_inter) // self.display_inter - 1 + 1])
        self.train_ave_loss_path = \
            np.zeros([1, ((self.num_sample - self.num_burnin) * self.thin_inter) // self.display_inter - 1 + 1])

        if not(x_val == []) and not(y_val == []):
            self.num_val = self.x_val.shape[0]
            self.val_predict_path = \
                np.zeros([self.num_val, ((self.num_sample - self.num_burnin) * self.thin_inter) // self.display_inter - 1 + 1])
            self.val_loss_path = \
                np.zeros([1, ((self.num_sample - self.num_burnin) * self.thin_inter) // self.display_inter - 1 + 1])
            self.val_ave_loss_path = \
                np.zeros([1, ((self.num_sample - self.num_burnin) * self.thin_inter) // self.display_inter - 1 + 1])
        else:
            self.val_predict_path = []
            self.val_loss_path = []
            self.val_ave_loss_path = []


        if not (x_test == []) and not (y_test == []):
            self.num_test = self.x_test.shape[0]
            self.test_predict_path = \
                np.zeros([self.num_test, ((self.num_sample - self.num_burnin) * self.thin_inter) // self.display_inter - 1 + 1])
            self.test_loss_path = \
                np.zeros([1, ((self.num_sample - self.num_burnin) * self.thin_inter) // self.display_inter - 1 + 1])
            self.test_ave_loss_path = \
                np.zeros([1, ((self.num_sample - self.num_burnin) * self.thin_inter) // self.display_inter - 1 + 1])
        else:
            self.test_predict_path = []
            self.test_loss_path = []
            self.test_ave_loss_path = []


        # ------------------------- Start Sampling ---------------------------- #
        np.random.seed(self.seed)
        with tf.Session() as sess:
            tf.set_random_seed(self.seed)
            tf.global_variables_initializer().run()

            # Burnin info
            sess.run(tf.assign(self.W1, self.ctd_W1))
            sess.run(tf.assign(self.W2, self.ctd_W2))
            sess.run(self.update_W, feed_dict={self.update_W1_value: self.ctd_Record_W1_N,
                                               self.update_W2_value: self.ctd_Record_W2_N})

            self.Record_W1_N = self.ctd_Record_W1_N
            self.Record_W2_N = self.ctd_Record_W2_N
            self.Record_noise_sigma = self.ctd_Record_noise_sigma
            self.noise_sigma_value = self.ctd_noise_sigma_value


            for iti in range(self.num_burnin * self.thin_inter, (self.num_sample + 1) * self.thin_inter):
                n1 = (iti * self.num_subsam) % self.num_train
                if n1 + self.num_subsam <= self.num_train:
                    n2 = n2 = n1 + self.num_subsam
                    subsam_range = np.arange(n1, n2)
                else:
                    subsam_range1 = np.arange(n1, self.num_train)
                    subsam_range2 = np.arange(0, ((iti + 1) * self.num_subsam) % self.num_train)
                    subsam_range = np.append(subsam_range1, subsam_range2)

                # -------------- Step 1: Gibbs sampler of noise_sigma ---------------- #

                if iti % self.gibbs_inter == 0:
                    temitirun = sess.run(self.loss, feed_dict={self.x: self.x_train, self.yt: self.y_train})

                    # error check
                    if -temitirun/self.num_train >= 1e8 or np.isnan(temitirun):
                        print('loss numeric error')
                        return

                    self.noise_sigma_value = \
                        self.Inverse_Gamma_Post(self.IG_prior_a, self.IG_prior_b, self.num_train, -temitirun, seed=self.seed)

                    if iti // self.thin_inter < self.num_N:
                        self.Record_noise_sigma[:, iti // self.thin_inter] = self.noise_sigma_value
                    else:
                        self.Record_noise_sigma[:, 1:self.num_N] = self.Record_noise_sigma[:, 0:self.num_N - 1]
                        self.Record_noise_sigma[:, 0] = self.noise_sigma_value

                # -------------------------- Step 2: update W -------------------------- #
                # If repulsive dynamics
                if self.lr2 != 0:
                    itirun = sess.run([self.train_step1, self.train_step2],
                                      feed_dict={self.noise_sigma: self.noise_sigma_value,
                                                 self.x: self.x_train[subsam_range, :],
                                                 self.yt: self.y_train[subsam_range, :],
                                                 self.lr1_hold: self.lr1, self.lr2_hold: self.lr2,
                                                 self.sqrtl1_hold: self.sqrtl1})


                    if iti % self.thin_inter == 0:
                        temitirun = sess.run([self.W1, self.W2])
                        # check numeric error
                        if np.isnan(temitirun[0]).any() or np.isnan(temitirun[1]).any() or not self.noise_sigma_value:
                            print('weight/sigma numeric error')
                            return
                            # raise ValueError('Numerical Error: Nan')

                        num_effect_sample = iti // self.thin_inter + 1
                        if num_effect_sample < self.num_N:
                            # Update Record_W1_N/Record_W2_N
                            num_effect_sample = iti // self.thin_inter + 1
                            self.Record_W1_N[[iti // self.thin_inter], :, :] = temitirun[0]
                            self.Record_W2_N[[iti // self.thin_inter], :, :] = temitirun[1]
                            num_rep = self.num_N // num_effect_sample + 1
                            tem_Record_W1_N = \
                                np.repeat(self.Record_W1_N[0:num_effect_sample, :, :], num_rep, axis=0)[0:self.num_N, :, :]
                            tem_Record_W2_N = \
                                np.repeat(self.Record_W2_N[0:num_effect_sample, :, :], num_rep, axis=0)[0:self.num_N, :, :]
                            sess.run(self.update_W, feed_dict={self.update_W1_value: tem_Record_W1_N,
                                                               self.update_W2_value: tem_Record_W2_N})
                            del tem_Record_W1_N
                            del tem_Record_W2_N

                        else:
                            self.Record_W1_N[1:self.num_N, :, :] = self.Record_W1_N[0:self.num_N - 1, :, :]
                            self.Record_W1_N[0, :, :] = temitirun[0]
                            self.Record_W2_N[1:self.num_N, :, :] = self.Record_W2_N[0:self.num_N - 1, :, :]
                            self.Record_W2_N[0, :, :] = temitirun[1]
                            sess.run(self.update_W, feed_dict={self.update_W1_value: self.Record_W1_N,
                                                               self.update_W2_value: self.Record_W2_N})
                # If not repulsive dynamics
                else:
                    itirun = sess.run([self.pretrain_step1, self.pretrain_step2],
                                      feed_dict={self.noise_sigma: self.noise_sigma_value,
                                                 self.x: self.x_train[subsam_range, :],
                                                 self.yt: self.y_train[subsam_range, :],
                                                 self.lr1_hold: self.lr1,
                                                 self.sqrtl1_hold:self.sqrtl1})

                    if iti % self.thin_inter == 0:
                        temitirun = sess.run([self.W1, self.W2])
                        if np.isnan(temitirun[0]).any() or np.isnan(temitirun[1]).any() or not self.noise_sigma_value:
                            print('weight/sigma numeric error')
                            return
                            # raise ValueError('Numerical Error: Nan')

                # ------------------------ Step 3: predict and loss --------------------------- #


                if iti // self.thin_inter > self.num_burnin and iti % self.display_inter == 0:

                    self.sigma_path[display_count] = self.noise_sigma_value

                    if not self.isnormalize:

                        itirun = sess.run([self.y, self.loss], feed_dict={self.x: self.x_train, self.yt: self.y_train})
                        self.train_predict_path[:, [display_count]] = itirun[0]
                        self.train_loss_path[:, display_count] = np.sqrt(-itirun[1] / self.num_train)
                        self.train_ave_loss_path[:, display_count] = \
                            np.sqrt(np.mean(np.square(self.y_train.T -
                                                    np.mean(self.train_predict_path[:, 0:display_count + 1], axis=1))))

                        if not (self.x_val == []) and not (self.y_val == []):
                            itirun = sess.run([self.y, self.loss], feed_dict={self.x: self.x_val, self.yt: self.y_val})
                            self.val_predict_path[:, [display_count]] = itirun[0]
                            self.val_loss_path[:, display_count] = np.sqrt(-itirun[1] / self.num_val)
                            self.val_ave_loss_path[:, display_count] = \
                                np.sqrt(np.mean(np.square(self.y_val.T -
                                                        np.mean(self.val_predict_path[:, 0:display_count + 1], axis=1))))

                        if not (self.x_test == []) and not (self.y_test == []):
                            itirun = sess.run([self.y, self.loss], feed_dict={self.x: self.x_test, self.yt: self.y_test})
                            self.test_predict_path[:, [display_count]] = itirun[0]
                            self.test_loss_path[:, display_count] = np.sqrt(-itirun[1] / self.num_test)
                            self.test_ave_loss_path[:, display_count] = \
                                np.sqrt(np.mean(np.square(self.y_test.T -
                                                        np.mean(self.test_predict_path[:, 0:display_count + 1], axis=1))))
                    else:
                        itirun = sess.run([self.y, self.loss_unnorm], feed_dict={self.x: self.x_train, self.yt: self.y_train,
                                                                                 self.std: self.y_train_std,
                                                                                 self.mean: self.y_train_mean})
                        self.train_predict_path[:, [display_count]] = itirun[0] * self.y_train_std + self.y_train_mean
                        self.train_loss_path[:, display_count] = \
                            np.sqrt(np.mean(np.square(self.train_predict_path[:, [display_count]].flatten() -
                                                      ((self.y_train.T*self.y_train_std)+self.y_train_mean))))
                        self.train_ave_loss_path[:, display_count] = \
                            np.sqrt(np.mean(np.square(((self.y_train.T*self.y_train_std)+self.y_train_mean) -
                                                      np.mean(self.train_predict_path[:, 0:display_count + 1], axis=1))))

                        if not (self.x_val == []) and not (self.y_val == []):
                            itirun = sess.run([self.y, self.loss_unnorm], feed_dict={self.x: self.x_val, self.yt: self.y_val,
                                                                                     self.std: self.y_train_std,
                                                                                     self.mean: self.y_train_mean})
                            self.val_predict_path[:, [display_count]] = itirun[0] * self.y_train_std + self.y_train_mean
                            self.val_loss_path[:, display_count] = np.sqrt(-itirun[1] / self.num_val)
                            self.val_ave_loss_path[:, display_count] = \
                                np.sqrt(np.mean(np.square(self.y_val.T - np.mean(self.val_predict_path[:, 0:display_count + 1], axis=1))))

                        if not (self.x_test == []) and not (self.y_test == []):
                            itirun = sess.run([self.y, self.loss_unnorm], feed_dict={self.x: self.x_test, self.yt: self.y_test,
                                                                                     self.std: self.y_train_std,
                                                                                     self.mean: self.y_train_mean})
                            self.test_predict_path[:, [display_count]] = itirun[0] * self.y_train_std + self.y_train_mean
                            self.test_loss_path[:, display_count] = np.sqrt(-itirun[1] / self.num_test)
                            self.test_ave_loss_path[:, display_count] = \
                                np.sqrt(np.mean(np.square(self.y_test.T - np.mean(self.test_predict_path[:, 0:display_count + 1], axis=1))))


                    print('iteration ', iti-self.num_burnin * self.thin_inter, '/', self.thin_inter * (self.num_sample - self.num_burnin))
                    print('noise_sigma =', self.noise_sigma_value)
                    print('Train')
                    print('train_loss =', self.train_loss_path[:, display_count])
                    print('train_aveloss=', self.train_ave_loss_path[:, display_count])
                    if not (self.x_val == []) and not (self.y_val == []):
                        print('Val')
                        print('val_loss =', self.val_loss_path[:, display_count])
                        print('val_aveloss=', self.val_ave_loss_path[:, display_count])
                    if not (self.x_test == []) and not (self.y_test == []):
                        print('Test')
                        print('test_loss =', self.test_loss_path[:, display_count])
                        print('test_aveloss=', self.test_ave_loss_path[:, display_count])
                    print('---------------------------------------------')
                    display_count += 1

                # ----------------------- Calculate the Magnituide of the gradient ---------------------- #
                if self.Grad_inter > 0 and iti % self.Grad_inter == 0:
                    itirun = sess.run([self.gradP, self.W1_gradK, self.W2_gradK],
                                   feed_dict={self.noise_sigma: self.noise_sigma_value,
                                              self.x: self.x_train[subsam_range, :],
                                              self.yt: self.y_train[subsam_range, :]})

                    W1_GradRatio = np.sqrt(np.mean(np.square(itirun[0][0]/itirun[1])))
                    W2_GradRatio = np.sqrt(np.mean(np.square(itirun[0][1]/itirun[2])))
                    print('W1: Grad Ratio of Langevin/SVGD = ', W1_GradRatio)
                    print('W2: Grad Ratio of Langevin/SVGD = ', W2_GradRatio)
                    print('---------------------------------------------')



    def Inverse_Gamma_Post(self, a, b, n, se, seed=1234):
        """
        -- Function for sampling noise variance based on conjugate prior --
        :param a: shape parameter of prior
        :param b: scale parameter of prior
        :param n: number of sample size
        :param se: sum square error
        :param seed: seed for random number
        :return:
        """
        # prior: z ~ IG(a,b) ~ z^-(a+1) * exp(-b/z)
        # posterior: z ~ IG(a+n/2, b+se/2); shape = a+n/2, scale = b+se/2
        np.random.seed(seed=seed)
        sigma = scipy.stats.invgamma.rvs(a + n / 2, scale=b + se / 2.0)
        return sigma



