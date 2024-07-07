import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from double_pendulum import DoublePendulumDataset

# Step 1: Initialize the double pendulum dataset with higher energy initial conditions
initial_condition = (np.pi / 2, 5.0, np.pi, 5.0)  # Larger initial angles and velocities
dataset = DoublePendulumDataset(max_time=10, dt=0.01, mass_1=1.0, mass_2=1.0, initial_condition=initial_condition)

# Convert the dataset to a DataFrame
df = dataset.get_df()

# Extract positions of the pendulum masses
x1 = df['x1'].values
x2 = df['x2'].values
y1 = df['y1'].values
y2 = df['y2'].values

# Step 2: Prepare figure for animation
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
line, = ax.plot([], [], 'o-', lw=2)
trail1, = ax.plot([], [], 'r-', alpha=0.5)
trail2, = ax.plot([], [], 'b-', alpha=0.5)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# Initialize animation
def init():
    line.set_data([], [])
    trail1.set_data([], [])
    trail2.set_data([], [])
    time_text.set_text('')
    return line, trail1, trail2, time_text

# Update animation frame
def update(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    trail1.set_data(x1[:i+1], y1[:i+1])
    trail2.set_data(x2[:i+1], y2[:i+1])
    time_text.set_text(time_template % (i * 0.01))
    return line, trail1, trail2, time_text

# Create animation
ani = FuncAnimation(fig, update, frames=len(x1), init_func=init, blit=True)

# Save animation as GIF
ani.save('output/double_pendulum_with_trails.gif', writer=PillowWriter(fps=30))

plt.show()
