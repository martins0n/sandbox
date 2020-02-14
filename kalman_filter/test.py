#%%
import kalman_filter_model
import numpy as np
from typing import Dict, Tuple, Sequence, List
from numpy.linalg import inv, multi_dot


#%%


class KalmanFilter:


    def __init__(self, 
            Y: np.ndarray,
            transisiton_matrix: np.ndarray,
            obseravation_matrix: np.ndarray,
            R_matrix: np.ndarray,
            Q_matrix: np.ndarray
            ) -> None:
        self.Y = Y
        self.transisiton_matrix = transisiton_matrix
        self.obseravation_matrix = obseravation_matrix
        self.initial_state = None
        self.initial_coavariance = None
        self.R_matrix = R_matrix
        self.Q_matrix = Q_matrix
    
    def filtering(self,
            initial_state: np.ndarray,
            initial_coavariance: np.ndarray
            ) -> np.ndarray:
        filtered_states = np.empty((
            len(self.Y),
            len(initial_state)
            ), dtype='float')
        filtered_states[0] = initial_state
        past_posterior_state = initial_state
        past_posterior_coavariance = initial_coavariance
        for i in range(1, len(self.Y)):
            past_posterior_state, past_posterior_coavariance =  self._forward_step(
                i, 
                past_posterior_state, 
                past_posterior_coavariance)
            filtered_states[i] = past_posterior_state
        
        return filtered_states
    
    def _forward_step(self,
            step: int,
            past_posterior_state: np.ndarray,
            past_posterior_coavariance: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:

        current_apriory_state = self.transisiton_matrix.dot(past_posterior_state)

        current_apriory_coavariance = multi_dot([
            self.transisiton_matrix, 
            past_posterior_coavariance,
            self.transisiton_matrix.T
        ]) + self.Q_matrix
    
        kalman_gain = multi_dot([
            current_apriory_coavariance,
            self.obseravation_matrix.T,
            inv(multi_dot([
                self.obseravation_matrix,
                current_apriory_coavariance,
                self.obseravation_matrix.T
                ]) + self.R_matrix
            )
        ])

        current_posterior_state = current_apriory_state \
            + kalman_gain.dot(self.Y[step]) \
            - multi_dot([
                kalman_gain,
                self.obseravation_matrix,
                current_apriory_state
            ])
      
        current_posterior_coavariance = current_apriory_coavariance \
            - multi_dot([
                kalman_gain,
                self.obseravation_matrix,
                current_apriory_coavariance
            ])
        return current_posterior_state, current_posterior_coavariance
#%%
Y =  np.array([[1],[3],[4],[4],[5]])
transisiton_matrix = np.array([
    [0.1, 0.9],
    [0.3, 0.3]])
obseravation_matrix = np.array([[1,0]])
R_matrix = np.array([[0.01]])
Q_matrix = np.array([
    [0.1, 0],
    [0, 0.1]])

#%%
kf = KalmanFilter(Y, 
transisiton_matrix, 
obseravation_matrix, R_matrix, Q_matrix)
#%%
kf.filtering(np.array([1,0]), np.array([
    [0.1, 0],
    [0, 0.1]]))
#%%
np.array([0,1]) + np.array([
    [0.1, 0],
    [0, 0.1]])
#%%
np.array([
    [0.1, 0],
    [0, 0.1],
    [0, 0.1]]).shape

#%%
import importlib
import kalman_filter_model
importlib.reload(kalman_filter_model)

#%%
class KalmanFilter:


    def __init__(self, 
            Y: np.ndarray,
            transisiton_matrix: np.ndarray,
            obseravation_matrix: np.ndarray,
            R_matrix: np.ndarray,
            Q_matrix: np.ndarray
            ) -> None:
        self.Y = Y
        self.transisiton_matrix = transisiton_matrix
        self.obseravation_matrix = obseravation_matrix
        self.initial_state = None
        self.initial_covariance = None
        self.R_matrix = R_matrix
        self.Q_matrix = Q_matrix
        self.list_posterior_covariance = list()
        self.list_apriory_covariance = list()
        self.list_posterior_state = list()
        self.list_apriory_state = list()
        self.list_smoothed_state = None
        self.list_smoothed_covariance = None

        
    def filtering(self,
            initial_state: np.ndarray,
            initial_covariance: np.ndarray
            ) -> Tuple[List, List]:
        filtered_states = np.empty((
            len(self.Y),
            len(initial_state)
            ), dtype='float')
        filtered_states[0] = initial_state
        self.list_posterior_state.append(initial_state)
        self.list_posterior_covariance.append(initial_covariance)
        for i in range(1, len(self.Y)):
            past_posterior_state, past_posterior_covariance =  self._forward_step(
                i, 
                self.list_posterior_state[-1], 
                self.list_posterior_covariance[-1])
            self.list_posterior_state.append(past_posterior_state)
            self.list_posterior_covariance.append(past_posterior_covariance)
        
        current_apriory_state = self.transisiton_matrix.dot(self.list_posterior_state[-1])
        self.list_apriory_state.append(current_apriory_state)
        
        current_apriory_covariance = multi_dot([
            self.transisiton_matrix, 
            self.list_posterior_covariance[-1],
            self.transisiton_matrix.T
        ]) + self.Q_matrix
        self.list_apriory_covariance.append(current_apriory_covariance)
        return self.list_posterior_state, self.list_posterior_covariance
    
    def _forward_step(self,
            step: int,
            past_posterior_state: np.ndarray,
            past_posterior_covariance: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:

        current_apriory_state = self.transisiton_matrix.dot(past_posterior_state)
        self.list_apriory_state.append(current_apriory_state)
        
        current_apriory_covariance = multi_dot([
            self.transisiton_matrix, 
            past_posterior_covariance,
            self.transisiton_matrix.T
        ]) + self.Q_matrix
        self.list_apriory_covariance.append(current_apriory_covariance)

        kalman_gain = multi_dot([
            current_apriory_covariance,
            self.obseravation_matrix.T,
            inv(multi_dot([
                self.obseravation_matrix,
                current_apriory_covariance,
                self.obseravation_matrix.T
                ]) + self.R_matrix
            )
        ])

        current_posterior_state = current_apriory_state \
            + kalman_gain.dot(self.Y[step]) \
            - multi_dot([
                kalman_gain,
                self.obseravation_matrix,
                current_apriory_state
            ])
      
        current_posterior_covariance = current_apriory_covariance \
            - multi_dot([
                kalman_gain,
                self.obseravation_matrix,
                current_apriory_covariance
            ])
        return current_posterior_state, current_posterior_covariance

    def smoothing(self):
        self.list_smoothed_state = list()
        self.list_smoothed_covariance = list()
        self.list_smoothed_covariance.append(self.list_posterior_covariance[-1])
        self.list_smoothed_state.append(self.list_posterior_state[-1])

        for t_step in range(len(self.Y), 1, -1):
            self._backward_step(t_step - 1)
        
        return self.list_smoothed_state, self.list_smoothed_covariance

    def _backward_step(self, t_step):
        
        L_matrix = multi_dot([
            self.list_apriory_covariance[t_step],
            self.transisiton_matrix.T,
            inv(self.list_posterior_covariance[t_step])
        ])
       

        print(self.list_smoothed_state[-1], self.list_posterior_state[t_step])
        self.list_smoothed_state.append(
            self.list_apriory_state[t_step] \
                + multi_dot([
                    L_matrix,
                    (self.list_smoothed_state[-1] - self.list_posterior_state[t_step])
                ])
        )

        self.list_smoothed_covariance.append(
            self.list_apriory_covariance[t_step] \
                + multi_dot([
                    L_matrix,
                    (self.list_smoothed_covariance[-1] - self.list_posterior_covariance[t_step]),
                    inv(L_matrix)
                ])
        )


#%%
np.random.seed(seed=40)
Y =  np.random.normal(size=(100,1))
transisiton_matrix = np.array([[1,1],[0,1]])
obseravation_matrix = np.array([[1,0]])
R_matrix = np.array([[1.0]])
Q_matrix = np.diag([0.1, 0.01])

 


#%%
kf = kalman_filter_model.KalmanFilter(Y, 
transisiton_matrix, 
obseravation_matrix, R_matrix, Q_matrix)
#%%
kf.filtering(np.array([1,0]), np.array([
    [0.5, 0],
    [0, 0.5]]))

b#%%
s = kf.smoothing()[0]
#%%
s
#%%
