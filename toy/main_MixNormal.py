import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.interactive(False)
import matplotlib.pyplot as plt
#import seaborn as sns
import importlib
import SRLD_Global as S
import pickle
import time
importlib.reload(S)

import torch
from torch_two_sample import statistics_diff
from torch.autograd import Variable
from scipy.stats import norm
from scipy.stats import wasserstein_distance

# ------------------------ parameter setting ----------------------- #
num_p = 2
num_N = 100
inter = 100
MaxLoop = 1000
num_rep = 20
SRLD = {}
SRLD_GradV = {}
SRLD_GradK = {}
LD = {}
LD_GradV = {}
IID = {}
MMD = {}
W = {}
TIME = np.zeros([2, num_rep])


# -------------------------- Experiment ---------------------------- #

np.random.seed(1234)
Dist_true = np.zeros([num_p, 2000])
for i in range(2000):
    tem = np.random.uniform(low=0, high=1)
    if tem <= 1.0/2.0:
        Dist_true[:, i] = np.random.standard_normal(size=num_p) + 1.0
    else:
        Dist_true[:, i] = np.random.standard_normal(size=num_p) - 1.0


# Parameter Setting:
# Use mean = 1.0 and std = 1.0 for initialization.

for i in range(num_rep):
    seed = 1234+i
    Sampler = S.SRLD(num_p, seed=seed)

    print('start iteration =', i)
    # Ours
    Langevinlr = 1e-3
    SVGDlr = 10e-3

    Sampler.Init(num_N)
    Sampler.SetSampler('Mixture Normal', lr1=Langevinlr, lr2=SVGDlr, h_factor=1.0)

    start = time.clock()
    res1, res1_GradV, res1_GradK = Sampler.RunSampler(MaxLoop=MaxLoop, inter=inter)
    end = time.clock()

    SRLD[i] = res1
    SRLD_GradV[i] = res1_GradV
    SRLD_GradK[i] = res1_GradK
    TIME[0, i] = end-start

    print('In iteration ', i, 'SRLD Finish', 'Time=', TIME[0, i])

    print(np.sqrt(np.mean(res1_GradV ** 2, axis=1)) / np.sqrt(np.mean(res1_GradK ** 2, axis=1)))
    # Base
    n1 = np.sqrt(np.mean(np.sum(np.square(Langevinlr * res1_GradV + SVGDlr * res1_GradK), axis=0)))
    n2 = np.sqrt(np.mean(np.sum(np.square(Langevinlr * res1_GradV), axis=0)))
    Adjustlr = Langevinlr * (n1 / n2)
    print('n1 =', n1)
    print('n2 =', n2)
    print(Adjustlr)
    #Adjustlr = Langevinlr + 1.0*SVGDlr/np.mean(np.sqrt(np.mean(res1_GradV ** 2, axis=1)) / np.sqrt(np.mean(res1_GradK ** 2, axis=1)))
    # Adjustlr = Langevinlr
    Sampler.Init(num_N)
    Sampler.SetSampler('Mixture Normal', lr1=Adjustlr, lr2=0.0)

    start = time.clock()
    res2, res2_GradV, res2_GradK = Sampler.RunSampler(MaxLoop=MaxLoop, inter=inter)
    end = time.clock()

    LD[i] = res2
    LD_GradV[i] = res2_GradV
    TIME[1, i] = end - start

    print('In iteration ', i, 'LD Finish', 'Time=', TIME[1, i])



    # Calculating MMD and Wasserstein
    Inter = 50
    temMMD1 = np.zeros((MaxLoop - num_N) // Inter)
    temMMD2 = np.zeros((MaxLoop - num_N) // Inter)
    temW1 = np.zeros((MaxLoop - num_N) // Inter)
    temW2 = np.zeros((MaxLoop - num_N) // Inter)

    for j in range((MaxLoop - num_N) // Inter):
        num_sample = 50*(j+1) # Sample size.
        fr_test = statistics_diff.MMDStatistic(num_sample, 2000)
        t_val_1, matrix_1 = fr_test(Variable(torch.FloatTensor(res1[:, num_N:num_N + num_sample].T)),
                                    Variable(torch.FloatTensor(Dist_true[:, :].T)), alphas=[.1], ret_matrix=True)
        t_val_2, matrix_2 = fr_test(Variable(torch.FloatTensor(res2[:, num_N:num_N + num_sample].T)),
                                    Variable(torch.FloatTensor(Dist_true[:, :].T)), alphas=[.1], ret_matrix=True)

        temMMD1[j] = t_val_1
        temMMD2[j] = t_val_2

        temW1[j] = wasserstein_distance(res1[0, num_N:num_N + num_sample].T, Dist_true[0,].T) + wasserstein_distance(
            res1[1, num_N:num_N + num_sample].T, Dist_true[1,].T)
        temW2[j] = wasserstein_distance(res2[0, num_N:num_N + num_sample].T, Dist_true[0,].T) + wasserstein_distance(
            res2[1, num_N:num_N + num_sample].T, Dist_true[1,].T)


    MMD[i] = np.matrix([temMMD1, temMMD2])
    W[i] = np.matrix([temW1, temW2])


    print('In iteration ', i, 'MMD Finish')
    np.savetxt('MMD'+str(i)+'.csv', MMD[i], delimiter=',')
    np.savetxt('W' + str(i) + '.csv', W[i], delimiter=',')
    np.savetxt('Time.csv', TIME, delimiter=',')

    Result = {'SRLD': SRLD, 'LD': LD, 'MMD': MMD, 'SRLD_GradV': SRLD_GradV, 'SRLD_GradK': SRLD_GradK,'LD_GradV': LD_GradV}
    #Result = {'SRLD': SRLD, 'LD': LD, 'MMD': MMD, 'W': W}
    Name = 'Result_MixNormal.pkl'
    output = open(Name, 'wb')
    pickle.dump(Result, output)
    output.close()