import matplotlib.pyplot as plt
import csv
import os
import time
import sys
import numpy as np

idx_start = 0
idx_end = 6
kin_or_dyn = 1
if len(sys.argv) == 2:
    idx_start = int(sys.argv[1])
elif len(sys.argv) == 3:
    idx_start = int(sys.argv[1])
    idx_end = int(sys.argv[2])
elif len(sys.argv) == 4:
    idx_start = int(sys.argv[1])
    idx_end = int(sys.argv[2])
    kin_or_dyn = int(sys.argv[3])


row_idx_to_be_ploted = range(idx_start,idx_end)
name = 'theta_sDDot' # theta_s or theta_sDDot
if kin_or_dyn == 0:
    name = 'theta_s'

iter_start = 1
directory = 'data/'

while 1:
    fig1 = plt.figure(1)
    ax1 = fig1.gca()

    for row_idx in row_idx_to_be_ploted:
        theta_i = []
        iteration = iter_start
        while os.path.isfile(directory+str(iteration)+'_'+name+'.csv'):
            matrix = np.genfromtxt (directory+str(iteration)+'_'+name+'.csv', delimiter=",")
            theta_i.append(matrix[row_idx])
            iteration+=1
        length = len(theta_i)
        t = range(iter_start,length+iter_start)
        ax1.plot(t,theta_i, label=''+name+'('+str(row_idx)+')')

    plt.xlabel('iterations')
    plt.ylabel('parameter values')
    plt.title(name)
    # plt.legend()
    plt.draw()

    plt.pause(60)
    plt.clf()
