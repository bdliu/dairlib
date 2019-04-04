import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import time
import sys


iter_start = 11
iter_end = 11
if len(sys.argv) == 2:
    iter_start = int(sys.argv[1])
if len(sys.argv) == 3:
    iter_start = int(sys.argv[1])
    iter_end = int(sys.argv[2])


batch_max = 5
min_dist = 0.24
delta_dist = 0.03

directory = 'data/'

while 1:
    fig1 = plt.figure(1)
    ax1 = fig1.gca()

    total_cost = []
    len_total_cost = 0;
    for batch in reversed(range(batch_max)):
        cost = []
        iteration = iter_start
        while os.path.isfile(directory+str(iteration)+'_'+str(batch)+'_c.csv'):
            # way1
            matrix = np.genfromtxt (directory+str(iteration)+'_'+str(batch)+'_c.csv', delimiter=",")
            cost.append(matrix)
            # way2
            # with open(directory+str(iteration)+'_'+str(batch)+'_c.csv','r') as csvfile:
            #     plots = csv.reader(csvfile, delimiter=',')
            #     for row in plots:
            #         cost.append(row[0])
            if iteration == iter_end:
                break;
            iteration+=1

        length = len(cost)
        t = range(iter_start,length+iter_start)
        ax1.plot(t,cost, label='stride length = '+str(min_dist+batch*delta_dist)+' (m)')

        # plot total cost
        if batch == batch_max-1:
            len_total_cost = len(cost)
            total_cost = cost
        else:
            print(batch)
            total_cost = [x + y for x, y in zip(total_cost, cost[0:len_total_cost])]
        if batch == 0:
            average_cost = [x/batch_max for x in total_cost]
            ax1.plot(t[0:len_total_cost],average_cost, 'k--', linewidth=2.0, label='Averaged cost')

    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.title('Cost over iterations')
    plt.legend()
    plt.draw()

    plt.pause(10)
    plt.clf()
