import matplotlib.pyplot as plt
import csv
import os
import time

col_ploted = range(6) #68 or 6
name = 'theta_sDDot' # theta_s or theta_sDDot

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
    plt.title('parameter')
    plt.legend()
    plt.draw()

    plt.pause(10)
    plt.clf()
