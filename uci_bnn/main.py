import dataset as BD
import numpy as np
import matplotlib
matplotlib.interactive(False)
import matplotlib.pyplot as plt
import importlib
import srld as S
import pickle
import time
import os

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='uci with bnn')

    # common
    parser.add_argument('--num_hidden', type=int, default=50, help='num of hidden units')
    parser.add_argument('--num_rep', type=int, default=20, help='num of repeated trials')

    # sampler
    parser.add_argument('--num_particle', type=int, default=10, help='num of particle')
    parser.add_argument('--batch_size', type=int, default=100, help='num of hidden units')
    parser.add_argument('--num_sample', type=int, default=500, help='num of sample to collect in total')
    parser.add_argument('--num_burnin', type=int, default=400, help='num of sample to discard as burn-in')

    # lr for sampler
    parser.add_argument('--burnin_lr', type=float, default=0.02, help='lr for burn-in')
    parser.add_argument('--srld_lr1', type=float, default=0.01, help='lr for Langevin gradient for srld')
    parser.add_argument('--srld_lr2', type=float, default=0.01, help='lr for repulsive gradient for srld (alpha is scaled into lr)')
    parser.add_argument('--ld_lr1', type=float, default=0.02, help='lr for Langevin dynamics')

    # freqs for sampler
    parser.add_argument('--thin', type=int, default=100, help='thinning factor (freq to collect the sample)')
    parser.add_argument('--gibbs_freq', type=int, default=25, help='freq to update sigma with gibbs sampler')
    parser.add_argument('--verbose', type=int, default=100, help='freq to store the result')
    parser.add_argument('--gred_freq', type=int, default=0, help='num of iteration to estimate the gradients ratio of Langevin/SVGD; if <=0: not estimate')

    args = parser.parse_args()


    Datalist = ['Energy']
    for data_name in Datalist:
        print('Data = ', data_name)

        train_loss = {}
        train_aveloss = {}
        test_loss = {}
        test_aveloss = {}
        train_predict_LD = {}
        train_predict_SRLD = {}
        test_predict_LD = {}
        test_predict_SRLD = {}
        train_set_x = {}
        train_set_y = {}
        test_set_x = {}
        test_set_y = {}
        sigma_path = {}
        TIME = np.zeros([2, args.num_rep])

        for i in range(args.num_rep):

            if data_name == 'Energy':
                base_burnin_lr1 = args.burnin_lr
                base_lr1_SRLD = args.srld_lr1
                base_lr2_SRLD = args.srld_lr2
                base_lr1_LD = args.ld_lr1

                FullData = BD.Energy()
                D = FullData.get_data(split=i)

            # Adding Bias
            Bias = 1
            x_train = D['X']
            y_train = D['Y']
            if Bias:
                x_train = np.column_stack((np.ones(x_train.shape[0]), x_train))

            x_test = D['Xs']
            y_test = D['Ys']
            if Bias:
                x_test = np.column_stack((np.ones(x_test.shape[0]), x_test))

            num_p = x_train.shape[1]
            num_train = x_train.shape[0]

            tem_train_loss = np.zeros([2, ((args.num_sample - args.num_burnin) * args.thin) // args.verbose - 1 + 1])
            tem_train_aveloss = np.zeros([2, ((args.num_sample - args.num_burnin) * args.thin) // args.verbose - 1 + 1])
            tem_test_loss = np.zeros([2, ((args.num_sample - args.num_burnin) * args.thin) // args.verbose - 1 + 1])
            tem_test_aveloss = np.zeros([2, ((args.num_sample - args.num_burnin) * args.thin) // args.verbose - 1 + 1])

            tem_sigma_path = np.zeros([2, ((args.num_sample - args.num_burnin) * args.thin) // args.verbose - 1 + 1])

            # Run sampler
            Sampler = S.SRLD_BNN(seed=1234 + i)

            Sampler.Init(num_p, args.num_hidden, args.num_particle)
            Sampler.SetSampler(num_train, num_subsam=args.batch_size)
            # Burnin
            burnin_start = time.clock()
            Sampler.RunBurnin(x_train=x_train, y_train=y_train,
                              base_burnin_lr1=base_burnin_lr1,
                              num_burnin=args.num_burnin,
                              thin_inter=args.thin,
                              gibbs_inter=args.gibbs_freq)

            burnin_end = time.clock()
            burnin_time = burnin_end - burnin_start

            # LD
            start = time.clock()
            Sampler.RunSampler(x_test=x_test, y_test=y_test,
                               base_lr1=base_lr1_LD, base_lr2=0,
                               num_sample=args.num_sample,
                               display_inter=args.verbose,
                               Grad_inter=args.gred_freq)
            end = time.clock()
            TIME[1, i] = end - start + burnin_time

            tem_train_loss[1, :] = Sampler.train_loss_path
            tem_train_aveloss[1, :] = Sampler.train_ave_loss_path
            tem_test_loss[1, :] = Sampler.test_loss_path
            tem_test_aveloss[1, :] = Sampler.test_ave_loss_path

            tem_sigma_path[1, :] = Sampler.sigma_path

            train_loss[i] = tem_train_loss
            train_aveloss[i] = tem_train_aveloss
            test_loss[i] = tem_test_loss
            test_aveloss[i] = tem_test_aveloss
            sigma_path[i] = tem_sigma_path

            train_predict_LD[i] = Sampler.train_predict_path
            test_predict_LD[i] = Sampler.test_predict_path
            # SRLD

            start = time.clock()
            Sampler.RunSampler(x_test=x_test, y_test=y_test,
                               base_lr1=base_lr1_SRLD, base_lr2=base_lr2_SRLD,
                               num_sample=args.num_sample,
                               display_inter=args.verbose,
                               Grad_inter=args.gred_freq)
            end = time.clock()
            TIME[0, i] = end - start + burnin_time

            tem_train_loss[0, :] = Sampler.train_loss_path
            tem_train_aveloss[0, :] = Sampler.train_ave_loss_path
            tem_test_loss[0, :] = Sampler.test_loss_path
            tem_test_aveloss[0, :] = Sampler.test_ave_loss_path

            tem_sigma_path[0, :] = Sampler.sigma_path

            train_loss[i] = tem_train_loss
            train_aveloss[i] = tem_train_aveloss
            test_loss[i] = tem_test_loss
            test_aveloss[i] = tem_test_aveloss
            sigma_path[i] = tem_sigma_path

            train_predict_SRLD[i] = Sampler.train_predict_path
            test_predict_SRLD[i] = Sampler.test_predict_path

            # ------------- train/test set -------------- #
            train_set_x[i] = Sampler.x_train
            train_set_y[i] = Sampler.y_train
            test_set_x[i] = Sampler.x_test
            test_set_y[i] = Sampler.y_test

            # ------------- Save ------------ #
            path_name = data_name
            folder = os.path.exists(path_name)
            if not folder:
                os.makedirs(path_name)

            np.savetxt(data_name + '/tem_train_loss' + str(i) + '.csv', train_loss[i], delimiter=',')
            np.savetxt(data_name + '/tem_train_aveloss' + str(i) + '.csv', train_aveloss[i], delimiter=',')
            np.savetxt(data_name + '/tem_test_loss' + str(i) + '.csv', test_loss[i], delimiter=',')
            np.savetxt(data_name + '/tem_test_aveloss' + str(i) + '.csv', test_aveloss[i], delimiter=',')
            np.savetxt(data_name + '/Time.csv', TIME, delimiter=',')

            Result = {'train_loss': train_loss, 'train_aveloss': train_aveloss, 'test_loss': test_loss,
                      'test_aveloss': test_aveloss,
                      'TIME': TIME,
                      'train_predict_SRLD': train_predict_SRLD, 'test_predict_SRLD': test_predict_SRLD,
                      'train_predict_LD': train_predict_LD, 'test_predict_LD': test_predict_LD,
                      'train_sigma_path': sigma_path,
                      'train_set_x': train_set_x, 'train_set_y': train_set_y, 'test_set_x': test_set_x,
                      'test_set_y': test_set_y}


            Name = data_name + '/Result_BNN_' + data_name + '.pkl'
            output = open(Name, 'wb')
            pickle.dump(Result, output)
            output.close()











