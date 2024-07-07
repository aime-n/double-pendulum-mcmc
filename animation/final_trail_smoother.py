import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from double_pendulum import DoublePendulumDataset

# Parameters to control the duration and speed of the GIF
max_time = 20  # Total simulation time in seconds
fps = 30  # Frames per second
interval = 100  # Interval between frames in milliseconds
dt = 0.02  # Time step for the simulation

# Step 1: Initialize the double pendulum dataset with higher energy initial conditions
initial_condition = (np.pi / 2, 5.0, np.pi, 5.0)  # Larger initial angles and velocities
dataset = DoublePendulumDataset(max_time=max_time, dt=dt, mass_1=1.0, mass_2=1.0, initial_condition=initial_condition)

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
line, = ax.plot([], [], 'o-', lw=2, color='black')

# Increase the number of trail segments for a smoother fading effect
trail_length = 30
trail1_lines = [ax.plot([], [], '-', color=(1, 0.6, 0.6, alpha))[0] for alpha in np.linspace(0.05, 1, trail_length)]
trail2_lines = [ax.plot([], [], '-', color=(0.6, 0.6, 1, alpha))[0] for alpha in np.linspace(0.05, 1, trail_length)]

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

# Initialize animation
def init():
    line.set_data([], [])
    for trail in trail1_lines:
        trail.set_data([], [])
    for trail in trail2_lines:
        trail.set_data([], [])
    time_text.set_text('')
    return [line, *trail1_lines, *trail2_lines, time_text]

# Update animation frame
def update(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    
    for j in range(trail_length):
        if i - j > 0:
            trail1_lines[j].set_data(x1[i-j:i+1], y1[i-j:i+1])
            trail2_lines[j].set_data(x2[i-j:i+1], y2[i-j:i+1])
    
    time_text.set_text(time_template % (i * dt))
    return [line, *trail1_lines, *trail2_lines, time_text]

# Create animation
ani = FuncAnimation(fig, update, frames=len(x1), init_func=init, blit=True, interval=interval)

# Save animation as GIF
ani.save('output/double_pendulum_with_smoother_fading_trail.gif', writer=PillowWriter(fps=fps))

plt.show()
