import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # Required for 3D projection

# Define the range for theta1 and theta2
theta1_vals = np.linspace(-3, 3, 100)
theta2_vals = np.linspace(-3, 3, 100)

# Create a meshgrid for theta1 and theta2
T1, T2 = np.meshgrid(theta1_vals, theta2_vals)

# Define the loss function with a stable minimum (Elliptic Paraboloid)
# L(theta1, theta2) = theta1^2 + theta2^2
Loss = T1**2 + T2**2

# Create the 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
# cmap options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'RdBu', etc.
surf = ax.plot_surface(T1, T2, Loss, cmap='viridis', edgecolor='none', alpha=0.9)

# Add labels and title
ax.set_xlabel('$\\theta_1$', fontsize=12)
ax.set_ylabel('$\\theta_2$', fontsize=12)
ax.set_zlabel('Loss Value', fontsize=12)
ax.set_title('Loss Function with Stable Minima: $L(\\theta_1, \\theta_2) = \\theta_1^2 + \\theta_2^2$', fontsize=14)

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.6, aspect=10, label='Loss Value')

# Adjust viewing angle for better visualization of the minimum (optional)
ax.view_init(elev=30, azim=-120) # Adjusted azimuth to better see the bowl shape

# You can also plot contour lines on the base plane to visualize the minimum
# The contour lines for L = theta1^2 + theta2^2 will be circles
cset_xy = ax.contour(T1, T2, Loss, zdir='z', offset=np.min(Loss) - 1, cmap="coolwarm", linewidths=1.5, levels=10)
# The 'offset' places the contours slightly below the surface.
# 'levels' determines how many contour lines are drawn.

# Enhance grid and panes for better 3D perception
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.grid(True, linestyle='--', alpha=0.7)

# Mark the minimum point (optional, but good for clarity)
min_point_t1 = 0
min_point_t2 = 0
min_loss = min_point_t1**2 + min_point_t2**2
ax.scatter([min_point_t1], [min_point_t2], [min_loss], color='red', s=100, label='Stable Minimum (0,0)', depthshade=True)
ax.legend(loc='upper left')


plt.show()
