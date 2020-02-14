import numpy as np
from typing import Dict, Tuple, Sequence, List
from numpy.linalg import inv, multi_dot


class KalmanFilter:
    def __init__(
        self,
        Y: np.ndarray,
        transisiton_matrix: np.ndarray,
        obseravation_matrix: np.ndarray,
        R_matrix: np.ndarray,
        Q_matrix: np.ndarray,
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

    def filtering(
        self, initial_state: np.ndarray, initial_covariance: np.ndarray
    ) -> Tuple[List, List]:

        self.list_posterior_state.append(initial_state)
        self.list_posterior_covariance.append(initial_covariance)
        for i in range(1, len(self.Y)):
            past_posterior_state, past_posterior_covariance = self._forward_step(
                i, self.list_posterior_state[-1], self.list_posterior_covariance[-1]
            )
            self.list_posterior_state.append(past_posterior_state)
            self.list_posterior_covariance.append(past_posterior_covariance)

        return self.list_posterior_state, self.list_posterior_covariance

    def _forward_step(
        self,
        step: int,
        past_posterior_state: np.ndarray,
        past_posterior_covariance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        current_apriory_state = self.transisiton_matrix.dot(past_posterior_state)
        self.list_apriory_state.append(current_apriory_state)

        current_apriory_covariance = (
            multi_dot(
                [
                    self.transisiton_matrix,
                    past_posterior_covariance,
                    self.transisiton_matrix.T,
                ]
            )
            + self.Q_matrix
        )
        self.list_apriory_covariance.append(current_apriory_covariance)

        kalman_gain = multi_dot(
            [
                current_apriory_covariance,
                self.obseravation_matrix.T,
                inv(
                    multi_dot(
                        [
                            self.obseravation_matrix,
                            current_apriory_covariance,
                            self.obseravation_matrix.T,
                        ]
                    )
                    + self.R_matrix
                ),
            ]
        )

        current_posterior_state = (
            current_apriory_state
            + kalman_gain.dot(self.Y[step])
            - multi_dot([kalman_gain, self.obseravation_matrix, current_apriory_state])
        )

        current_posterior_covariance = current_apriory_covariance - multi_dot(
            [kalman_gain, self.obseravation_matrix, current_apriory_covariance]
        )
        return current_posterior_state, current_posterior_covariance

    def smoothing(self):
        self.list_smoothed_state = list()
        self.list_smoothed_covariance = list()
        self.list_smoothed_covariance.append(self.list_posterior_covariance[-1])
        self.list_smoothed_state.append(self.list_posterior_state[-1])

        for t_step in range(len(self.Y) - 1, 0, -1):
            self._backward_step(t_step - 1)

        return self.list_smoothed_state, self.list_smoothed_covariance

    def _backward_step(self, t_step):
        L_matrix = multi_dot(
            [
                self.list_posterior_covariance[t_step],
                self.transisiton_matrix.T,
                inv(self.list_apriory_covariance[t_step]),
            ]
        )

        self.list_smoothed_state.insert(
            0,
            self.list_posterior_state[t_step]
            + multi_dot(
                [
                    L_matrix,
                    (self.list_smoothed_state[0] - self.list_apriory_state[t_step]),
                ]
            ),
        )

        self.list_smoothed_covariance.insert(
            0,
            self.list_posterior_covariance[t_step]
            + multi_dot(
                [
                    L_matrix,
                    (
                        self.list_smoothed_covariance[0]
                        - self.list_apriory_covariance[t_step]
                    ),
                    L_matrix.T,
                ]
            ),
        )
