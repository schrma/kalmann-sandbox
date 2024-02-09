import numpy as np
from scipy.linalg import expm
from kfsims.pendulum import run_sim
from kfsims.kfmodels import KalmanFilterBase

# Simulation Options
sim_options = {'time_step': 0.01,
               'end_time': 10,
               'measurement_rate': 10,
               'measurement_noise_std': np.deg2rad(0.1),
               'start_at_random_angle': True,
               'draw_plots': True,
               'draw_animation': True} 

kf_options =  {'torque_std':0.1, # Q Matrix
              'meas_std':np.deg2rad(0.1), # R Matrix  
              'init_pos_std':np.deg2rad(10),
              'init_vel_std':np.deg2rad(10),
              'init_on_measurement':True}

# Kalman Filter Model
class KalmanFilterModel(KalmanFilterBase):

    
    def initialise(self, time_step, torque_std, meas_std, length = 0.5, init_on_measurement=False, init_pos_std = 0.1, init_vel_std = 0.1):
        self.dt = time_step
        self.init_pos_std = init_pos_std
        self.init_vel_std = init_vel_std
        
        # Set Model F and H Matrices
        A = np.array([[0,1],[-9.81/length,0]])
        self.F = expm(A*time_step) # Note: use the expm() function
        self.H = np.array([[1,0]]) # Note: use np.array([[a,b]]) for a row matrix

        # Set R and Q Matrices
        self.Q = np.diag(np.array([0,1]) * (torque_std*torque_std)) # Note: use variable 'torque_std'
        self.R = meas_std*meas_std # Note: use variable 'meas_std'

        # Set Initial State and Covariance 
        if init_on_measurement is False:
            self.state = np.transpose(np.array([[0,0]])) # Assume we are at zero position and velocity
            self.covariance = np.diag(np.array([init_pos_std*init_pos_std,init_vel_std*init_vel_std]))
        
        return
    
    def prediction_step(self):
        # Make Sure Filter is Initialised
        if self.state is not None:
            x = self.state
            P = self.covariance

            # Calculate Kalman Filter Prediction
            x_predict = np.matmul(self.F, x) 
            P_predict = np.matmul(self.F, np.matmul(P, np.transpose(self.F))) + self.Q

            # Save Predicted State
            self.state = x_predict
            self.covariance = P_predict

        return

    def update_step(self, measurement):

        # Make Sure Filter is Initialised
        if self.state is not None and self.covariance is not None:
            x = self.state
            P = self.covariance
            H = self.H
            R = self.R

            # Calculate Kalman Filter Update
            z = measurement
            z_hat = x[0]
            y = z - z_hat
            S = np.matmul(H,np.matmul(P,np.transpose(H))) + R
            K = np.matmul(P,np.matmul(np.transpose(H),np.linalg.inv(S)))
            x_update = x + K*y # Since y is a scalar, we can't use np.matmul(K, y)
            P_update = np.matmul( (np.eye(2) - np.matmul(K,H)), P)

            # Save Updated State
            self.innovation = y
            self.innovation_covariance = S
            self.state = x_update
            self.covariance = P_update

        else:

            # Set Initial State and Covariance 
            self.state = np.transpose(np.array([[measurement,0]]))
            self.covariance = np.diag(np.array([self.R,self.init_vel_std*self.init_vel_std])) # Assume we don't know our velocity

        return 


# Run the Simulation
run_sim(KalmanFilterModel, sim_options, kf_options)