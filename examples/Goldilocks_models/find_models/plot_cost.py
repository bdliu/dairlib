import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import time
import sys

only_plot_average_cost = True

iter_start = 1
iter_end = 11
is_iter_end = 0
if len(sys.argv) == 2:
    iter_start = int(sys.argv[1])
if len(sys.argv) == 3:
    iter_start = int(sys.argv[1])
    iter_end = int(sys.argv[2])
    is_iter_end = 1


n_sampel_sl = 3  # should be > 0
n_sampel_gi = 3  # should be > 0
batch_max = n_sampel_sl * n_sampel_gi
min_dist = 0.3 - 0.015
delta_dist = 0.015
min_incline = -0.05
delta_incline = 0.05


directory = 'data/'

# file_name = 'c.csv'
file_name = 'c_without_tau.csv'

while 1:
    fig1 = plt.figure(1)
    ax1 = fig1.gca()

    total_cost = []
    len_total_cost = 0;
    for batch in reversed(range(batch_max)):
        cost = []
        iteration = iter_start
        while os.path.isfile(directory+str(iteration)+'_'+str(batch)+'_'+file_name):
            # way1
            matrix = np.genfromtxt (directory+str(iteration)+'_'+str(batch)+'_'+file_name, delimiter=",")
            cost.append(matrix)
            # way2
            # with open(directory+str(iteration)+'_'+str(batch)+'_'+file_name,'r') as csvfile:
            #     plots = csv.reader(csvfile, delimiter=',')
            #     for row in plots:
            #         cost.append(row[0])
            if is_iter_end & (iteration == iter_end):
                break;
            iteration+=1

        length = len(cost)
        t = range(iter_start,length+iter_start)
        if not only_plot_average_cost:
            if n_sampel_gi > 1:
                ax1.plot(t,cost, label='stride length = '+str(min_dist+(batch%n_sampel_sl)*delta_dist)+' (m), ground incline = '+str(min_incline+(batch/n_sampel_gi)*delta_incline)+' (rad)')
            else:
                ax1.plot(t,cost, label='stride length = '+str(min_dist+(batch%n_sampel_sl)*delta_dist)+' (m)')

        # plot total cost
        if batch == batch_max-1:
            len_total_cost = len(cost)
            total_cost = cost
        else:
            total_cost = [x + y for x, y in zip(total_cost, cost[0:len_total_cost])]
        if batch == 0:
            average_cost = [x/batch_max for x in total_cost]
            if only_plot_average_cost:
                ax1.plot(t[0:len_total_cost],average_cost, 'b--', linewidth=2.0, label='Averaged cost')
            else:
                ax1.plot(t[0:len_total_cost],average_cost, 'k--', linewidth=2.0, label='Averaged cost')

    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    if not only_plot_average_cost:
        plt.title('Cost over iterations')
        plt.legend()
    plt.draw()

    plt.pause(10)
    plt.clf()