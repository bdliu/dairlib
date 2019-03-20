import matplotlib.pyplot as plt
import csv
import os
import time

batch_max = 5

while 1:
    fig1 = plt.figure(1)
    ax1 = fig1.gca()

    for batch in range(batch_max):
        cost = []
        iteration = 1
        while os.path.isfile('data/'+str(iteration)+'_'+str(batch)+'_c.csv'):
            with open('data/'+str(iteration)+'_'+str(batch)+'_c.csv','r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')
                for row in plots:
                    cost.append(row[0])
            iteration+=1
        length = len(cost)
        t = range(1,length+1)
        ax1.plot(t,cost, label='stride length = '+str(0.24+batch*0.03)+' (m)')

    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.title('Cost over iterations')
    plt.legend()
    plt.draw()

    plt.pause(10)
    plt.clf()
