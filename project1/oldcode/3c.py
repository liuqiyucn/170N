import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd

# Animation for earth
# initialization

# getting planet data analytic
data_ana = pd.read_csv('ana.csv')
earth_ana = data_ana[data_ana.planet_label == 2]
earth_ana = earth_ana.to_numpy()
# getting planet data simulation
data_sim = pd.read_csv('sim.csv')
earth_sim = data_sim[data_sim.planet_label == 2]

x = earth_sim['r_x'].values
y = earth_sim['r_y'].values
z = earth_sim['r_z'].values

# Create the figure and axes for the plot
fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

# Create the initial point in the plot
point, = ax.plot(x[0], y[0], z[0], 'bo')

ax.plot(earth_ana[:,2], earth_ana[:,3], earth_ana[:,4], color = "red", label = "Analytic")

# Define the update function for the animation
def update(frame):
    # Update the position of the point
    point.set_data(x[frame:frame+1], y[frame:frame+1])
    point.set_3d_properties(z[frame:frame+1])
    return point,

# Create the animation object
anim = FuncAnimation(fig, update, frames = len(x), blit=True)

ax.set_xlabel("x (au)")
ax.set_ylabel("y (au)")
ax.set_zlabel("z (au)")
ax.set_title('Earth Animation')
# Show the plot
plt.legend()
plt.show()