import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.interactive(False)
import matplotlib.pyplot as plt
#import seaborn as sns
import importlib
import srld as S
import pickle
import time
importlib.reload(S)

import torch
from torch_two_sample import statistics_diff
from torch.autograd import Variable
from scipy.stats import norm
import pyhmc
from pyhmc import hmc

# ------------------------ parameter setting ----------------------- #
num_p = 2
num_N = 100
inter = 1
MaxLoop = 2000
num_rep = 20
SRLD = {}
SRLD_GradV = {}
SRLD_GradK = {}
LD = {}
LD_GradV = {}
ESS = np.zeros([2, num_rep])
TIME = np.zeros([2, num_rep])
MMD = {}
# ------------------------ Experiment --------------------------- #

np.random.seed(1)
def logprob(x):
    a1 = 10.0
    a2 = 4.0
    b1 = 1.2
    a3 = 1.0
    a4 = 2.0
    logp = -x[0]**4/a1 - (a2*(x[1]+b1) - a3*x[0]**2)**2/a4
    grad = np.array([-4.0/a1*x[0]**3 - 2.0/a4*(a2*(x[1]+b1) - a3*x[0]**2)*(-2.0*a3*x[0]),
          -2.0 / a4 * (a2 * (x[1] + b1) - a3 * x[0] ** 2) * a2])
    return logp, grad

Dist_true = hmc(logprob, x0=np.random.randn(2), n_samples=int(2e3))
Dist_true = Dist_true.T

for i in range(num_rep):
    seed = 1234+i
    Sampler = S.SRLD(num_p, seed=seed)

    print('start iteration =', i)

    # ---------------------- lr 0.5e-2
    # Ours
    Langevinlr = 1e-2
    SVGDlr = 1e-1
    Sampler.Init(num_N)
    Sampler.SetSampler('Rosenbrock', lr1=Langevinlr, lr2=SVGDlr, h_factor=1.0)

    start = time.clock()
    res1, res1_GradV, res1_GradK = Sampler.RunSampler(MaxLoop=MaxLoop, inter=inter)
    end = time.clock()

    SRLD[i] = res1
    SRLD_GradV[i] = res1_GradV
    SRLD_GradK[i] = res1_GradK
    TIME[0, i] = end-start

    print('In iteration ', i, 'SRLD Finish', 'Time=', TIME[0, i])
    print(np.sqrt(np.mean(res1_GradV ** 2, axis=1)) / np.sqrt(np.mean(res1_GradK ** 2, axis=1)))

    #Adjustlr = Langevinlr + 1.0*SVGDlr/np.mean(np.sqrt(np.mean(res1_GradV ** 2, axis=1)) / np.sqrt(np.mean(res1_GradK ** 2, axis=1)))
    n1 = np.sqrt(np.mean(np.sum(np.square(Langevinlr*res1_GradV + SVGDlr*res1_GradK),axis=0)))
    n2 = np.sqrt(np.mean(np.sum(np.square(Langevinlr*res1_GradV), axis=0)))
    Adjustlr = Langevinlr*(n1/n2)
    print('n1 =', n1)
    print('n2 =', n2)
    print(Adjustlr)
    print(Langevinlr + 1.0 * SVGDlr / np.mean(np.sqrt(np.mean(res1_GradV ** 2, axis=1)) / np.sqrt(np.mean(res1_GradK ** 2, axis=1))))
    #Adjustlr = Langevinlr + 1.0 * SVGDlr / np.mean(np.sqrt(np.mean(res1_GradV ** 2, axis=1)) / np.sqrt(np.mean(res1_GradK ** 2, axis=1)))

    # Base lr = 2
    Sampler.Init(num_N)
    Sampler.SetSampler('Rosenbrock', lr1=Adjustlr, lr2=0.0)

    start = time.clock()
    res2, res2_GradV, res2_GradK = Sampler.RunSampler(MaxLoop=MaxLoop, inter=inter)
    end = time.clock()

    LD[i] = res2
    LD_GradV[i] = res2_GradV
    TIME[1, i] = end - start

    print('In iteration ', i, 'LD Finish', 'Time=', TIME[1, i])

    # Calculating MMD
    Inter = 100
    temMMD1 = np.zeros((MaxLoop - num_N) // Inter)
    temMMD2 = np.zeros((MaxLoop - num_N) // Inter)


    for j in range((MaxLoop - num_N) // Inter):
        num_sample = Inter*(j+1)  # Sample size.
        fr_test = statistics_diff.MMDStatistic(num_sample, 2000)
        t_val_1, matrix_1 = fr_test(Variable(torch.FloatTensor(res1[:, num_N:num_N + num_sample].T)),
                                    Variable(torch.FloatTensor(Dist_true[:, :].T)), alphas=[.1], ret_matrix=True)
        t_val_2, matrix_2 = fr_test(Variable(torch.FloatTensor(res2[:, num_N:num_N + num_sample].T)),
                                    Variable(torch.FloatTensor(Dist_true[:, :].T)), alphas=[.1], ret_matrix=True)

        temMMD1[j] = t_val_1
        temMMD2[j] = t_val_2

    MMD[i] = np.matrix([temMMD1, temMMD2])

    print('In iteration ', i, 'MMD Finish')
    np.savetxt('MMD_Rosen'+str(i)+'.csv', MMD[i], delimiter=',')
    np.savetxt('Time_Rosen.csv', TIME, delimiter=',')

    Result = {'SRLD': SRLD, 'LD': LD, 'MMD': MMD, 'SRLD_GradV': SRLD_GradV, 'SRLD_GradK': SRLD_GradK, 'LD_GradV': LD_GradV}
    Name = 'Result_Rosen.pkl'
    output = open(Name, 'wb')
    pickle.dump(Result, output)
    output.close()

