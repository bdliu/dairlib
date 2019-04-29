import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import time
import sys


iteration_start = 1
iteration_end = 35
if len(sys.argv) == 2:
    iteration_start = int(sys.argv[1])
elif len(sys.argv) == 3:
    iteration_start = int(sys.argv[1])
    iteration_end = int(sys.argv[2])


batch = 0
directory = 'data/'

# Get the max and min
max_list = []
min_list = []
for iteration in range(iteration_start,iteration_end+1):
    if os.path.isfile(directory+str(iteration)+'_'+str(batch)+'_t_and_s.csv'):
        matrix = np.genfromtxt (directory+str(iteration)+'_'+str(batch)+'_t_and_s.csv', delimiter=",")
        n_rows = (matrix.shape)[0]

        for index in range(1,n_rows):
            s_i_th_element = matrix[index,:]
            max_list.append(np.amax(s_i_th_element))
            min_list.append(np.amin(s_i_th_element))
max_val = max(max_list)
min_val = min(min_list)

# Plot
fig = plt.figure(1)

for iteration in range(iteration_start,iteration_end+1):
    ax = fig.gca()
    if os.path.isfile(directory+str(iteration)+'_'+str(batch)+'_t_and_s.csv'):
        matrix = np.genfromtxt (directory+str(iteration)+'_'+str(batch)+'_t_and_s.csv', delimiter=",")
        n_rows = (matrix.shape)[0]

        for index in range(1,n_rows):
            t = matrix[0,:]
            s_i_th_element = matrix[index,:]
            ax.plot(t,s_i_th_element, label='s_'+str(index))
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)

    cost = []
    if os.path.isfile(directory+str(iteration)+'_'+str(batch)+'_c.csv'):
        matrix = np.genfromtxt (directory+str(iteration)+'_'+str(batch)+'_c.csv', delimiter=",")
        cost.append(matrix)

    plt.xlabel('t (seconds)', fontsize=15)
    plt.ylabel('s', fontsize=15)
    # plt.title('Generalized position trajectories.')
    plt.title('Iteration #'+str(iteration)+': cost = '+str(cost[0]))
    leg = plt.legend()


    # Set the axis limit
    # ax.set_xlim(0, 0.95)
    ax.set_ylim(min_val, max_val)

    # Draw the figure so you can find the positon of the legend.
    plt.draw()

    # # Get the bounding box of the original legend
    # bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    # # Change to location of the legend.
    # bb.x0 += 0
    # # bb.x1 += 0.12
    # leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

    if (iteration == iteration_end):
        plt.show()
    else:
        plt.draw()
        plt.pause(1)
        plt.clf()