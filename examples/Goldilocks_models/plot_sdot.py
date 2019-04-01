import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import time


iteration_start = 35
iteration_end = 35

iteration = 1
batch = 0
directory = 'data/'

fig = plt.figure(1)

for iteration in range(iteration_start,iteration_end+1):
    ax = fig.gca()
    if os.path.isfile(directory+str(iteration)+'_'+str(batch)+'_t_and_ds.csv'):
        matrix = np.genfromtxt (directory+str(iteration)+'_'+str(batch)+'_t_and_ds.csv', delimiter=",")
        n_rows = (matrix.shape)[0]

        for index in range(1,n_rows):
            t = matrix[0,:]
            ds_i_th_element = matrix[index,:]
            ax.plot(t,ds_i_th_element, label='ds_'+str(index))
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)

    cost = []
    if os.path.isfile(directory+str(iteration)+'_'+str(batch)+'_c.csv'):
        matrix = np.genfromtxt (directory+str(iteration)+'_'+str(batch)+'_c.csv', delimiter=",")
        cost.append(matrix)

    plt.xlabel('t (seconds)', fontsize=15)
    plt.ylabel('ds', fontsize=15)
    # plt.title('Generalized position trajectories.')
    plt.title('Iteration #'+str(iteration)+': cost = '+str(cost[0]))
    leg = plt.legend()


    # Set the axis limit
    # ax.set_xlim(0, 0.95)
    # ax.set_ylim(-0.45, 1.1)

    # Draw the figure so you can find the positon of the legend.
    plt.draw()

    # # Get the bounding box of the original legend
    # bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    # # Change to location of the legend.
    # bb.x0 += 0
    # # bb.x1 += 0.12
    # leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

    if (iteration_start == iteration_end):
        plt.show()
    else:
        plt.draw()
        plt.pause(1)
        plt.clf()
