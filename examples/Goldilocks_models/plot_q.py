import matplotlib.pyplot as plt
import csv
import os
import time

min_dist = 0.24
delta_dist = 0.03

iteration = 19
batch = 2
directory = 'data/'

fig1 = plt.figure(1)
ax1 = fig1.gca()

t = []
if os.path.isfile(directory+str(iteration)+'_'+str(batch)+'_time_at_knots.csv'):
    with open(directory+str(iteration)+'_'+str(batch)+'_time_at_knots.csv','r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            t.append(row[0])
for joint in range(7):
    joint_angle = []
    if os.path.isfile(directory+str(iteration)+'_'+str(batch)+'_state_at_knots.csv'):
        with open(directory+str(iteration)+'_'+str(batch)+'_state_at_knots.csv','r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            row_num = 0
            for row in plots:
                if row_num == joint:
                    joint_angle = row;
                row_num+=1
    ax1.plot(t,joint_angle, label='q('+str(joint)+')')

plt.xlabel('t (s)')
plt.ylabel('q (m or rad)')
plt.title('Generalized position trajectories')
plt.legend()
plt.show()

