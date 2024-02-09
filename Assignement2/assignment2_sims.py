import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import expm

# Simulation Properties
dt = 0.01
end_time = 10

# System Properties
g = 9.81
l = 0.5
theta0 = np.deg2rad(45)

# Linear Model
A = np.array([[0,1],[-g/l,0]])
F = expm(A*dt)

# Set Initial State
x0 = np.array([theta0,0])

# Save Initial State
x_nonlin = x0
x_lin = x0
nonlin_state_history = [x_nonlin]
lin_state_history = [x_lin]

# Run Simulation
num_steps = np.ceil(end_time/dt).astype(int)
for k in range(1,num_steps+1):

    # Linear
    x_lin = np.matmul(F,x_lin)

    # Non-Linear
    theta_ddot = -g/l*np.sin(x_nonlin[0])
    theta_dot = x_nonlin[1] + theta_ddot * dt
    theta = x_nonlin[0] + theta_dot * dt
    x_nonlin = np.array([theta,theta_dot])

    # Save State
    nonlin_state_history.append(x_nonlin)
    lin_state_history.append(x_lin)


# Plot Animation
fig1 = plt.figure(constrained_layout=True)
fig_ax = fig1.add_subplot(111,title='Pendulum Position', aspect='equal',xlim=(-1, 1), ylim=(-1.1, 0))
fig_ax.grid(True)
nonlin_mass_plot, = fig_ax.plot([], [], 'bo')
nonlin_arm_plot, = fig_ax.plot([],[],'b-')
lin_mass_plot, = fig_ax.plot([], [], 'ro')
lin_arm_plot, = fig_ax.plot([],[],'r-')

def update_plot(i):
    nonlin_theta = nonlin_state_history[i][0]
    lin_theta = lin_state_history[i][0]
    nonlin_mass_plot.set_data([np.sin(nonlin_theta),-np.cos(nonlin_theta)])
    nonlin_arm_plot.set_data([0,np.sin(nonlin_theta)],[0,-np.cos(nonlin_theta)])
    lin_mass_plot.set_data([np.sin(lin_theta),-np.cos(lin_theta)])
    lin_arm_plot.set_data([0,np.sin(lin_theta)],[0,-np.cos(lin_theta)])
    return nonlin_mass_plot, nonlin_arm_plot, lin_mass_plot, lin_arm_plot

# # Create the Animation
plot_animation = animation.FuncAnimation(fig1, update_plot, frames=range(0,num_steps,4),interval=1, repeat=False, blit=False)

# Show Animation
plt.show()