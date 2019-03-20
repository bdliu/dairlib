import matplotlib.pyplot as plt
import csv
import os
import time

batch_max = 5
min_dist = 0.24
delta_dist = 0.03

iter_start = 11
directory = 'data/'

while 1:
    fig1 = plt.figure(1)
    ax1 = fig1.gca()

    for batch in range(batch_max):
        cost = []
        iteration = iter_start
        while os.path.isfile(directory+str(iteration)+'_'+str(batch)+'_c.csv'):
            with open(directory+str(iteration)+'_'+str(batch)+'_c.csv','r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                for row in plots:
                    cost.append(row[0])
            iteration+=1
        length = len(cost)
        t = range(iter_start,length+iter_start)
        ax1.plot(t,cost, label='stride length = '+str(min_dist+batch*delta_dist)+' (m)')

    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.title('Cost over iterations')
    plt.legend()
    plt.draw()

    plt.pause(10)
    plt.clf()
