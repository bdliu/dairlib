import matplotlib.pyplot as plt
import csv
import os
import time
import sys

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


col_ploted = range(idx_start,idx_end)
name = 'theta_sDDot' # theta_s or theta_sDDot
if kin_or_dyn == 0:
    name = 'theta_s'

iter_start = 1
directory = 'data/'

while 1:
    fig1 = plt.figure(1)
    ax1 = fig1.gca()

    for col in col_ploted:
        theta_i = []
        iteration = iter_start
        while os.path.isfile(directory+str(iteration)+'_'+name+'.csv'):
            with open(directory+str(iteration)+'_'+name+'.csv','r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                row_num = 0
                for row in plots:
                    if row_num == col:
                        theta_i.append(row[0])
                    row_num+=1
            iteration+=1
        length = len(theta_i)
        t = range(iter_start,length+iter_start)
        ax1.plot(t,theta_i, label=''+name+'('+str(col)+')')

    plt.xlabel('iterations')
    plt.ylabel('parameter values')
    plt.title(name)
    # plt.legend()
    plt.draw()

    plt.pause(10)
    plt.clf()
