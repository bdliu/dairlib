import matplotlib.pyplot as plt
import csv
import os
import time

min_dist = 0.24
delta_dist = 0.03

iteration = 18
batch = 2
directory = 'data/'

fig = plt.figure(1)
ax = fig.gca()

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
    ax.plot(t,joint_angle, label='q('+str(joint)+')')
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

plt.xlabel('t (s)', fontsize=15)
plt.ylabel('q (m or rad)', fontsize=15)
# plt.title('Generalized position trajectories')
leg = plt.legend()


# Set the axis limit
ax.set_xlim(0, 0.95)
ax.set_ylim(-0.45, 1.1)

# Draw the figure so you can find the positon of the legend.
plt.draw()

# Get the bounding box of the original legend
bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

# Change to location of the legend.
bb.x0 += 0
# bb.x1 += 0.12
leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

plt.show()
