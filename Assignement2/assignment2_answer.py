import numpy as np
from kfsims.pendulum import run_sim
from kfsims.kfpendulum import KalmanFilterModel

# Simulation Options
sim_options = {'time_step': 0.01,
               'end_time': 10,
               'measurement_rate': 10,
               'measurement_noise_std': np.deg2rad(0.1),
               'start_at_random_angle': True,
               'draw_plots': True,
               'draw_animation': True} 

kf_options =  {'torque_std':0.01, # Q Matrix
              'meas_std':np.deg2rad(0.1), # R Matrix  
              'init_on_measurement':True}

# Run the Simulation
run_sim(KalmanFilterModel, sim_options, kf_options)