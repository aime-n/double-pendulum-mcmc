import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from double_pendulum import DoublePendulumDataset

# Load the dataset
dataset = DoublePendulumDataset()
train_loader, test_loader = dataset.get_train_test_data_loaders(batch_size=100)

# Example of data
print('Example of data: ', dataset[0])

# Pendulum parameters
L1, L2 = 1, 1
r = 0.05  # Plotted bob circle radius

# Function to convert polar coordinates to Cartesian coordinates
def get_cartesian_coords(theta1, theta2):
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

# Create a figure
fig, ax = plt.subplots()
ax.set_xlim(-L1 - L2 - r, L1 + L2 + r)
ax.set_ylim(-L1 - L2 - r, L1 + L2 + r)
ax.set_aspect('equal', adjustable='box')
plt.axis('off')

# Initialize the plot elements
rod, = ax.plot([], [], lw=2, c='k')
c0 = Circle((0, 0), r/2, fc='k', zorder=10)
c1 = Circle((0, 0), r, fc='b', ec='b', zorder=10)
c2 = Circle((0, 0), r, fc='r', ec='r', zorder=10)
ax.add_patch(c0)
ax.add_patch(c1)
ax.add_patch(c2)

def init():
    rod.set_data([], [])
    return rod, c1, c2

def update(frame):
    ax.clear()
    theta1, theta2 = frame
    x1, y1, x2, y2 = get_cartesian_coords(theta1, theta2)
    rod, = ax.plot([0, x1, x2], [0, y1, y2], lw=2, c='k')
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1, y1), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2, y2), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.set_xlim(-L1 - L2 - r, L1 + L2 + r)
    ax.set_ylim(-L1 - L2 - r, L1 + L2 + r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    return rod, c1, c2

# Extract the angles from the dataset
angles = [(item[0][0], item[0][2]) for item in dataset]

# Sample every 10th frame to reduce the number of frames
sampled_angles = angles[::10]

# Create the animation
ani = FuncAnimation(fig, update, frames=sampled_angles, init_func=init, blit=True)

# Save the animation as a GIF
if not os.path.exists('output'):
    os.makedirs('output')
ani.save('output/double_pendulum.gif', writer=PillowWriter(fps=10))

plt.show()
