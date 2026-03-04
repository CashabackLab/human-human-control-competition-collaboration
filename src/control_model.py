from copy import deepcopy
import numpy as np
import src.model_functions as mf
from scipy.integrate import solve_ivp
from numpy.linalg import pinv
from dataclasses import dataclass


@dataclass(kw_only=True)
class Dynamics:
    """
    Should house A,B,C,x0,I
    """

    A: np.ndarray
    A_probe: np.ndarray
    B1: np.ndarray
    C1: np.ndarray
    x0: np.ndarray
    N: int
    state_mapping: dict[str, int]
    sensory_delay: int = 0
    h: float = 0.01
    isDiscretized: bool = False
    isAugmented: bool = False

    def __post_init__(self):
        self.nx_original = len(self.state_mapping)

    @property
    def T(self):
        return self.N * self.h

    def discretize(self):
        if self.isDiscretized:
            print("Dynamics already discretized")
        else:
            self.A[: self.nx_original, : self.nx_original] = (
                np.eye(self.nx_original)
                + self.h * self.A[: self.nx_original, : self.nx_original]
            )
            self.A_probe[: self.nx_original, : self.nx_original] = (
                np.eye(self.nx_original)
                + self.h * self.A_probe[: self.nx_original, : self.nx_original]
            )
            self.B1 = self.B1 * self.h
            self.isDiscretized = True

    def undiscretize(self):
        if not self.isDiscretized:
            print("Dynamics are already continuous")
        else:
            self.A[: self.nx_original, : self.nx_original] = (
                self.A[: self.nx_original, : self.nx_original]
                - np.eye(self.nx_original)
            ) / self.h
            self.A_probe[: self.nx_original, : self.nx_original] = (
                self.A_probe[: self.nx_original, : self.nx_original]
                - np.eye(self.nx_original)
            ) / self.h
            self.B1 = self.B1 / self.h
            self.isDiscretized = False

    def augment(self):
        if not self.isAugmented:
            self.A = mf.augment_A_matrix(self.A, self.sensory_delay)
            self.A_probe = mf.augment_A_matrix(self.A_probe, self.sensory_delay)
            self.B1 = mf.augment_B_matrix(self.B1, self.sensory_delay)
            self.C1 = np.block(
                [
                    [
                        np.zeros(
                            (self.C1.shape[0], self.C1.shape[1] * (self.sensory_delay))
                        ),
                        self.C1,
                    ]
                ]
            )
            self.x0 = np.tile(self.x0, (self.sensory_delay + 1, 1))
            self.isAugmented = True


@dataclass(kw_only=True)
class GTODynamics(Dynamics):
    """
    As on B2 and C2
    """

    B2: np.ndarray
    C2: np.ndarray

    def discretize(self):
        if self.isDiscretized:
            print("Dynamics already discretized")
        else:
            super().discretize()
            self.B2 = self.h * self.B2

    def undiscretize(self):
        if not self.isDiscretized:
            print("Dynamics are already continuous")
        else:
            super().undiscretize()
            self.B2 = self.B2 / self.h

    def augment(self):
        if not self.isAugmented:
            super().augment()
            self.B2 = mf.augment_B_matrix(self.B2, self.sensory_delay)
            self.C2 = np.block(
                [
                    [
                        np.zeros(
                            (self.C2.shape[0], self.C2.shape[1] * (self.sensory_delay))
                        ),
                        self.C2,
                    ]
                ]
            )


@dataclass(kw_only=True)
class ControlPolicy:
    """
    Should house Q,R
    Needs Dynamics
    """

    dynamics: Dynamics | GTODynamics
    Q_self: np.ndarray
    R_self: np.ndarray
    isDiscretized: bool = False

    @property
    def A(self):
        return self.dynamics.A

    @property
    def B(self):
        return self.dynamics.B1

    @property
    def Q(self):
        return self.Q_self

    @property
    def R(self):
        return self.R_self

    @property
    def N(self):
        return self.dynamics.N

    def discretize(self):
        if self.isDiscretized:
            print("Policy already discretized")
        else:
            self.isDiscretized = True
            self.Q_self[:-1] = (
                self.Q_self[:-1] * self.dynamics.h
            )  # DO NOT DISCRETIZE FINAL Q
            self.R_self = self.R_self * self.dynamics.h

    def undiscretize(self):
        if not self.isDiscretized:
            print("Policy already continuous")
        else:
            self.isDiscretized = False
            self.Q_self[:-1] = self.Q_self[:-1] / self.dynamics.h
            self.R_self = self.R_self / self.dynamics.h

    def solve_ricatti_discrete(self):
        self.P = np.ones((self.N + 1, self.A.shape[0], self.A.shape[0])) * np.nan

        self.P[-1, :, :] = self.Q[
            -1
        ]  #! Set "first" P to be the last Q, this Q is NOT euler'd
        for i in reversed(range(self.N)):
            self.P[i, :, :] = (
                (self.A.T @ self.P[i + 1] @ self.A)
                - (self.A.T @ self.P[i + 1] @ self.B)
                @ pinv(self.R + self.B.T @ self.P[i + 1] @ self.B)
                @ (self.B.T @ self.P[i + 1] @ self.A)
                + self.Q[i]
            )

    def solve_ricatti_continuous(self):
        def dPdt(t, P):
            Pmat = P.reshape(*self.dynamics.A.shape)
            # Find the index for the current time for Q
            t_idx = int(round(t / self.dynamics.h))
            t_idx = min(t_idx, self.N)  # Clamp to valid range

            P_sol = -(
                self.A.T @ Pmat
                + Pmat @ self.A
                - (Pmat @ self.B) @ pinv(self.R) @ (self.B.T @ Pmat)
                + self.Q[t_idx]  # Use Q at current time
            )

            return P_sol.flatten()

        sol = solve_ivp(
            dPdt,
            t_span=(self.dynamics.T, 0),
            y0=self.Q[-1].flatten(),
            t_eval=np.linspace(self.dynamics.T, 0, self.N + 1),
            method="RK45",
            # max_step=1e-4
            # rtol=1e-6,  # Relative tolerance
            # atol=1e-9,  # Absolute tolerance
        )

        if not sol.success:
            print(f"Warning: solve_ivp failed with message: {sol.message}")
        # Flip P back around to get F
        self.P = np.array(
            [sol.y[:, i].reshape(*self.A.shape) for i in reversed(range(len(sol.t)))]
        )

    def set_feedback_gain_continuous(self):
        self.F = pinv(self.R) @ (self.B.T @ self.P[1:])

    def set_feedback_gain_discrete(self):
        self.F = pinv(self.R + self.B.T @ self.P[1:] @ self.B) @ (
            self.B.T @ self.P[1:] @ self.A
        )


@dataclass(kw_only=True)
class GTOControlPolicy(ControlPolicy):
    """ """

    player: int
    Q_partner: np.ndarray
    R_self_partner: np.ndarray
    R_partner_self: np.ndarray
    R_partner: np.ndarray
    alpha: float
    partner_knowledge: bool = True
    isAugmented: bool = False

    @property
    def Q1(self):
        return self.Q_self

    @property
    def Q2(self):
        return self.Q_partner

    @property
    def R11(self):
        return self.R_self

    @property
    def R12(self):
        return self.R_self_partner

    @property
    def R21(self):
        return self.R_partner_self

    @property
    def R22(self):
        return self.R_partner

    @property
    def B1(self):
        return self.get_B_matrices_self_partner()[0]

    @property
    def B2(self):
        return self.get_B_matrices_self_partner()[1]

    def discretize(self):
        if self.isDiscretized:
            print("policy already discretized")
        else:
            # We don't discretize the final cost
            self.Q_self[:-1] = self.Q_self[:-1] * self.dynamics.h
            self.Q_partner[:-1] = self.Q_partner[:-1] * self.dynamics.h

            # R is supposed to be divided by sampling period
            self.R_self = self.R_self * self.dynamics.h
            self.R_partner = self.R_partner * self.dynamics.h
            self.R_self_partner = self.R_self_partner * self.dynamics.h
            self.R_partner_self = self.R_partner_self * self.dynamics.h
            self.isDiscretized = True

    def undiscretize(self):
        if not self.isDiscretized:
            print("policy already continuous")
        else:
            # We don't discretize the final cost
            self.Q_self[:-1] = self.Q_self[:-1] / self.dynamics.h
            self.Q_partner[:-1] = self.Q_partner[:-1] / self.dynamics.h

            # R is supposed to be divided by sampling period
            self.R_self = self.R_self / self.dynamics.h
            self.R_partner = self.R_partner / self.dynamics.h
            self.R_self_partner = self.R_self_partner / self.dynamics.h
            self.R_partner_self = self.R_partner_self / self.dynamics.h
            self.isDiscretized = False

    def augment(self):
        if not self.isAugmented:
            self.isAugmented = True
            self.Q_self = mf.augment_Q_matrix(self.Q_self, self.dynamics.sensory_delay)
            self.Q_partner = mf.augment_Q_matrix(
                self.Q_partner, self.dynamics.sensory_delay
            )

    def care_about_partner_Q(self):
        """
        Two Targets: Since there are now two targets (rtx and ltx), you need to split the Q in half so the controller doesn't overshoot
          - Self.Alpha can be between [0,0.5] where 0 is I only care about my own target and 0.5 is I care equally about both targets
          - This is effectively choosing a new target center to aim for (0.5 would be between the two targets if they split off)
          - ?? Should you split the self.alpha on other features of the Q matrix other than the targets?
        """
        print("considering partner q with alpha:", self.alpha)
        state_mapping = self.dynamics.state_mapping
        ccx_id = state_mapping["ccx"]
        ccy_id = state_mapping["ccy"]
        # Can change this later, but I don't ever want to care about my partner's target more than my own
        assert self.alpha <= 0.5

        if self.player == 1:
            # Care about other person's target
            self_target_idx = state_mapping["rtx"]  
            self_target_idy = state_mapping["rty"]  
            partner_target_idx = state_mapping["ltx"]  
            partner_target_idy = state_mapping["lty"]  
        else:
            # Care about other person's target
            self_target_idx = state_mapping["ltx"]  
            self_target_idy = state_mapping["lty"]  
            partner_target_idx = state_mapping["rtx"]   
            partner_target_idy = state_mapping["rty"]  

        # Diagonal center cursor
        #! THIS WOULD BE AN ERROR... the F_ccx + (-F_rtx) + (-F_ltx) need to a up to ZERO to hit the target
        #! SINCE THERE ARE 2 targets that would a up, we do NOT want to cut the F_ccx by (1-alpha)...
        # self.Q_self[ccx_id,ccx_id] = (1-self.alpha)*self.Q_self[ccx_id,ccx_id] # Need to take the proportion off of my Q
        # self.Q_self[ccy_id,ccy_id] = (1-self.alpha)*self.Q_self[ccy_id,ccy_id] # Need to take the proportion off of my Q

        # Diagonal term
        ## How much of MY Q am I willing to give up for them
        self.Q_self[..., partner_target_idx, partner_target_idx] = (
            self.alpha * self.Q_self[..., self_target_idx, self_target_idx]
        )  
        # Need to take the proportion off of my Q
        self.Q_self[..., self_target_idx, self_target_idx] = (
            (1 - self.alpha) * self.Q_self[..., self_target_idx, self_target_idx]
        )  
        # How much of MY Q am I willing to give up for them
        self.Q_self[..., partner_target_idy, partner_target_idy] = (
            self.alpha * self.Q_self[..., self_target_idy, self_target_idy]
        ) 
        # Need to take the proportion off of my Q
        self.Q_self[..., self_target_idy, self_target_idy] = (
            (1 - self.alpha) * self.Q_self[..., self_target_idy, self_target_idy]
        )  

        # Off Diagonal term
        self.Q_self[..., ccx_id, partner_target_idx] = (
            self.alpha * self.Q_self[..., ccx_id, self_target_idx]
        )
        self.Q_self[..., ccx_id, self_target_idx] = (
            (1 - self.alpha) * self.Q_self[..., ccx_id, self_target_idx]
        )

        self.Q_self[..., ccy_id, partner_target_idy] = (
            self.alpha * self.Q_self[..., ccy_id, self_target_idy]
        )
        self.Q_self[..., ccy_id, self_target_idy] = (
            (1 - self.alpha) * self.Q_self[..., ccy_id, self_target_idy]
        )

        # Off Diagonal Term
        self.Q_self[..., partner_target_idx, ccx_id] = (
            self.alpha * self.Q_self[..., self_target_idx, ccx_id]
        )
        self.Q_self[..., self_target_idx, ccx_id] = (
            (1 - self.alpha) * self.Q_self[..., self_target_idx, ccx_id]
        )

        self.Q_self[..., partner_target_idy, ccy_id] = (
            self.alpha * self.Q_self[..., self_target_idy, ccy_id]
        )
        self.Q_self[..., self_target_idy, ccy_id] = (
            (1 - self.alpha) * self.Q_self[..., self_target_idy, ccy_id]
        )

    def get_B_matrices_self_partner(self):
        if self.player == 1:
            B1 = self.dynamics.B1  # Self B
            B2 = self.dynamics.B2  # Partner B
        elif self.player == 2:
            B1 = self.dynamics.B2  # If player ==2, then self B is B2
            B2 = self.dynamics.B1
        else:
            raise ValueError("self.player can only be 1 or 2")
        return B1, B2

    def solve_ricatti_discrete(self):
        assert self.isDiscretized, "Policy matrices is not discretized"
        assert self.dynamics.isDiscretized, "Dynamics matrices are not discretized"
        if self.dynamics.isAugmented:
            self.augment()
        # * Set matrices
        A = self.dynamics.A
        B1, B2 = self.get_B_matrices_self_partner()
        R11 = self.R_self
        R22 = self.R_partner
        R12 = self.R_self_partner
        R21 = self.R_partner_self
        N = self.dynamics.N

        # * Empty arrays for ricatti solution (P) and feedback gains (F)
        S = np.zeros((N, *A.shape))  # storing A - sum(B@P)
        self.P = np.zeros((N + 1, *A.shape))
        self.P_partner = np.zeros((N + 1, *A.shape))
        self.F = np.zeros((N, B1.shape[1], B1.shape[0]))
        self.F_partner = np.zeros((N, B2.shape[1], B2.shape[0]))

        self.P[-1] = self.Q_self[-1]
        # Keeps P2 as zeros, assuming partner applies no control
        if (self.partner_knowledge): 
            self.P_partner[-1] = self.Q_partner[-1]

        # NOTE Looping multiple times to handle the fact the F will be initially zeros for each F_partner solution
        for _ in range(10):
            for i in reversed(range(N)):
                # Normal feedback gain part
                # NOTE Calculating F2 first so that F1 can use it... we do this for both p1 and p2 bc we use the function twice
                self.F_partner[i] = pinv(R22 + B2.T @ self.P_partner[i + 1] @ B2) @ (
                    B2.T @ self.P_partner[i + 1] @ A
                    - B2.T @ self.P_partner[i + 1] @ B1 @ self.F[i]
                )
                self.F[i] = (
                    pinv(R11 + B1.T @ self.P[i + 1] @ B1)
                    @ (
                        B1.T @ self.P[i + 1] @ A
                        - B1.T @ self.P[i + 1] @ B2 @ self.F_partner[i]
                    )
                )  # note similarity for ofc feedback gains, only difference is B1.T@P1@B2@F2

                # Ricatti solution
                S[i] = A - B1 @ self.F[i] - B2 @ self.F_partner[i]
                self.P[i] = (
                    S[i].T @ self.P[i + 1] @ S[i]
                    + self.F[i].T @ R11 @ self.F[i]
                    + self.F_partner[i].T @ R12 @ self.F_partner[i]
                    + self.Q_self[i]
                )
                if self.partner_knowledge:  # Keeps P2 as zeroes if no partner knowledge
                    self.P_partner[i] = (
                        S[i].T @ self.P_partner[i + 1] @ S[i]
                        + self.F_partner[i].T @ R22 @ self.F_partner[i]
                        + self.F[i].T @ R21 @ self.F[i]
                        + self.Q_partner[i]
                    )

    def solve_ricatti_continuous(self):
        assert not self.isDiscretized
        assert not self.dynamics.isDiscretized
        if self.dynamics.isAugmented:
            self.augment()

        # * Set matrices
        A = self.dynamics.A
        B1, B2 = self.get_B_matrices_self_partner()
        Q1, Q2 = self.Q_self, self.Q_partner
        R11 = self.R_self
        R22 = self.R_partner
        R12 = self.R_self_partner
        R21 = self.R_partner_self

        if self.player == 1:
            assert B1 is self.dynamics.B1
        else:
            assert B1 is self.dynamics.B2

        def dPdt(t, P):
            """
            From Papavassilopoulos et al. (1979)
            "On the existence of Nash strategies and solutions to coupled riccati equations in linear-quadratic games"

            and Engwerda LQ Dynamic Optimization book
            """
            P1 = P[: A.shape[0] ** 2]  # First half of flattened initial solution is P1
            P2 = P[A.shape[0] ** 2 :]  # Second half of flattened initial solution is p2
            P1m = P1.reshape(A.shape[0], A.shape[0])  # Reshape into matrix form
            P2m = P2.reshape(A.shape[0], A.shape[0])

            if not self.partner_knowledge:
                P2m = np.zeros_like(P2m)

            t_idx = int(round(t / self.dynamics.h))
            t_idx = min(t_idx, self.dynamics.N)  # Clamp to valid range

            P1sol = -(
                P1m @ A
                + A.T @ P1m
                + Q1[t_idx]
                - P1m @ B1 @ pinv(R11) @ B1.T @ P1m
                - P1m @ B2 @ pinv(R22) @ B2.T @ P2m
                - P2m @ B2 @ pinv(R22) @ B2.T @ P1m
                + P2m @ B2 @ pinv(R22) @ R12 @ pinv(R22) @ B2.T @ P2m
            ).flatten()
            P2sol = -(
                P2m @ A
                + A.T @ P2m
                + Q2[t_idx]
                - P2m @ B2 @ pinv(R22) @ B2.T @ P2m
                - P2m @ B1 @ pinv(R11) @ B1.T @ P1m
                - P1m @ B1 @ pinv(R11) @ B1.T @ P2m
                + P1m @ B1 @ pinv(R11) @ R21 @ pinv(R11) @ B1.T @ P1m
            ).flatten()
            sol = np.hstack((P1sol, P2sol))
            return sol

        # * This reflects knowledge of the other players Q, not necessarily caring about it
        # * hstacked bc it's split up into P1 and P2 in the function
        QN = np.hstack((self.Q_self[-1].flatten(), self.Q_partner[-1].flatten()))

        P1_solution = solve_ivp(
            dPdt,
            t_span=(self.dynamics.T, 0),
            y0=QN,  # First P values
            t_eval=np.linspace(self.dynamics.T, 0, self.N + 1),
            method="RK45",
            # max_step=1e-4,
        )
        if P1_solution.status == -1:
            print(P1_solution)
            raise (ValueError("solution failed"))

        self.P = np.array(
            [
                P1_solution.y[: A.shape[0] ** 2, i].reshape(A.shape[0], A.shape[0])
                for i in reversed(range(len(P1_solution.t)))
            ]
        )
        self.P_partner = np.array(
            [
                P1_solution.y[A.shape[0] ** 2 :, i].reshape(A.shape[0], A.shape[0])
                for i in reversed(range(len(P1_solution.t)))
            ]
        )

        assert np.all(self.P[-1] == self.Q_self[-1])
        assert np.all(self.P_partner[-1] == self.Q_partner[-1])

    def set_feedback_gain_continuous(self):
        self.F = pinv(self.R11) @ self.B1.T @ self.P[1:]
        self.F_partner = pinv(self.R22) @ self.B2.T @ self.P_partner[1:]

    def set_feedback_gain_discrete(self):
        print("discrete GTO feedback gain is solved within solve_ricatti_discrete")
        pass
        # self.F = pinv(self.R11 + self.B1.T @ self.P[1:] @ self.B1) @ (self.B1.T @ self.P[1:] @ self.A)
        # self.F_partner = pinv(self.R22 + self.B2.T @ self.P_partner[1:] @ self.B2) @ (self.B2.T @ self.P_partner[1:] @ self.A)


@dataclass(kw_only=True)
class StateEstimator:
    """
    Should house W
    Needs Dynamics
    """

    dynamics: Dynamics | GTODynamics
    player: int
    W: np.ndarray
    V: np.ndarray
    sensor_noise_arr: np.ndarray
    internal_model_noise_arr: np.ndarray
    dual: bool = True
    LQR: bool = False
    isAugmented = False

    def __post_init__(self):
        assert self.internal_model_noise_arr.ndim == 2

    def augment(self) -> None:
        if not self.isAugmented:
            self.V = mf.augment_Q_matrix(
                self.V, sensory_delay=self.dynamics.sensory_delay
            )
            self.internal_model_noise_arr = mf.augment_B_matrix(
                self.internal_model_noise_arr, sensory_delay=self.dynamics.sensory_delay
            )
            self.isAugmented = True

    def get_observation_matrix(self) -> np.ndarray:
        if self.player == 1:
            C = self.dynamics.C1
        elif self.player == 2:
            C = self.dynamics.C2
        else:
            raise ValueError("player must be 1 or 2")
        return C

    def get_B_matrix(self) -> np.ndarray:
        if self.player == 1:
            B = self.dynamics.B1
        elif self.player == 2:
            B = self.dynamics.B2
        else:
            raise ValueError("player must be 1 or 2")
        return B

    def get_B_matrices_self_partner(self) -> tuple[np.ndarray, np.ndarray]:
        if self.player == 1:
            B1 = self.dynamics.B1  # Self B
            B2 = self.dynamics.B2  # Partner B
        elif self.player == 2:
            B1 = self.dynamics.B2  # If player ==2, then self B is B2
            B2 = self.dynamics.B1
        else:
            raise ValueError("self.player can only be 1 or 2")
        return B1, B2

    def set_linear_kalman_gain(self):
        """
        A: nx X nx
        - State transition matrix
        C: nz X nx
        - Observation Matrix
        W: nz X nz
        - Sensor/measurement covariance
        V: nx X nx
        - Process covariance
        N:
        - number of timesteps
        """
        assert self.dynamics.isDiscretized
        if self.dynamics.isAugmented:
            self.augment()

        A = self.dynamics.A
        N = self.dynamics.N
        C = self.get_observation_matrix()
        nx = A.shape[0]
        nz = C.shape[0]

        ## Calculate the optimal Kalman gain (forward in time) ##
        # Intialize
        self.P_prior = np.zeros(
            (N + 1, nx, nx)
        )  # Prior covariance !! CANNOT BE *np.nan, matrix multiplication gets weir
        self.P_post = np.zeros((N + 1, nx, nx)) * np.nan
        self.G = np.zeros((N + 1, nz, nz)) * np.nan  # State innnovation
        self.K = np.zeros((N + 1, nx, nz)) * np.nan  # Kalman gain
        I = np.eye(self.K[0].shape[0])  # identity matrix

        # Initial process covariance is the W value, this makes it so we prefer info from the first measurements
        self.P_prior[0, -nz:, -nz:] = self.W

        for i in range(N + 1):
            self.G[i, :, :] = (
                C @ self.P_prior[i, :, :] @ C.T + self.W
            )  # Covariance based on observation matrix and prior covariance + measurement covariance
            self.K[i, :, :] = (
                self.P_prior[i, :, :] @ C.T @ np.linalg.inv(self.G[i, :, :])
            )  # Optimal Kalman gain for current timestep
            # self.P_post[i,:,:] = (I - self.K[i,:,:] @ C) @ self.P_prior[i,:,:] # Updated covariance estimation, unstable version don't think it's a problem tho
            self.P_post[i, :, :] = (I - self.K[i, :, :] @ C) @ self.P_prior[i, :, :] @ (
                I - self.K[i, :, :] @ C
            ).T + self.K[i] @ self.W @ self.K[
                i
            ].T  # Updated covariance estimation using stable algorithm
            if i != N:
                self.P_prior[i + 1, :, :] = (
                    A @ self.P_post[i, :, :] @ A.T + self.V
                )  # Next prior covariance estimation is the current posterior

    @property
    def sensor_noise(self):
        return (
            np.zeros_like(self.sensor_noise_arr)[:, np.newaxis]
            if self.LQR
            else np.random.normal(0, self.sensor_noise_arr).squeeze()[:, np.newaxis]
        )

    @property
    def prediction_noise(self):
        shape = (self.nu, 1)
        ans = (
            np.zeros(shape)
            if self.LQR
            else np.random.normal(0, self.internal_model_noise_arr, shape)
        )
        return ans[self.nx :]  # just noise on current prediction

    def _get_prediction(self, prev_x, u1, u2=None):
        B1, B2 = self.get_B_matrices_self_partner()
        if self.dual:
            return (
                self.dynamics.A @ prev_x + B1 @ u1 + B2 @ u2
            )  # + self.internal_model_noise_arr
        else:
            return self.dynamics.A @ prev_x + B1 @ u1 + self.internal_model_noise_arr

    def get_estimate(
        self,
        t,
        measurement,
        prior,
        u_self,
        u_partner,
        partner_knowledge,
    ):
        nx = self.dynamics.A.shape[0]
        C = self.get_observation_matrix()

        # Set u2 to zero if no partner knowledge
        if not partner_knowledge:
            u_partner = np.zeros_like(u_self)

        # Sensory observation
        y_obs = C @ measurement + self.sensor_noise

        # Forward model prediction
        x_prediction = self._get_prediction(prior, u_self, u_partner)

        # Augment prediction state
        x_pred_aug = self._augment_state(
            x_pred=x_prediction[:nx],
            prev_x_aug=prior,  # prior comes from posterior estimate in run_dual_simulation
            num_states=nx,
        )  # Using the current x_prediction, but C can't observe this yet

        # Get posterior estimate
        x_post_next = x_pred_aug + self.K[t + 1] @ (y_obs - C @ x_pred_aug)

        return y_obs, x_pred_aug, x_post_next

    def _augment_state(self, x_pred, prev_x_aug, num_states):
        """
        Here we want to keep the prior states (previous posteriors), but then create the augmented prediction that uses
        those previous posterior states, then tacks on the new x_pred to the beginning

        That way, the observation matrix (which is [0,0,0,1]; crudely) only uses the LAST index of the states,
        which is the delayed posterior estimates
        """
        x_aug = deepcopy(prev_x_aug)  # Copy previous augmented state
        x_aug[num_states:] = prev_x_aug[
            :-num_states
        ]  # last part of x_aug is equal to first part of last augmented state (shifting down)
        x_aug[:num_states] = (
            x_pred  # Most up to date x is beginning of indices (meaning that C doesn't observe it)
        )
        return x_aug


@dataclass(kw_only=True)
class ModelSimulator:
    model_name: str
    movement_type: str
    dynamics: Dynamics | GTODynamics
    p1_policy: ControlPolicy | GTOControlPolicy
    p1_state_estimator: StateEstimator
    process_noise_value: float
    perturbation_onset_distance: float
    perturbation_states: list[str]
    perturbation_distances: list[float]
    probe_duration: int

    def __post_init__(self):
        assert (
            self.p1_state_estimator.dual == False
        )  # Must be false for model with single controller acting
        assert self.movement_type in ["reaching", "posture"], (
            "movement_type must be 'reaching' or 'posture'"
        )
        self.partner_knowledge = self.p1_policy.partner_knowledge

    def run_simulation(self):
        pass


@dataclass(kw_only=True)
class DualModelSimulator(ModelSimulator):
    p2_policy: ControlPolicy | GTOControlPolicy
    p2_state_estimator: StateEstimator

    def __repr__(self):
        return f"{self.model_name} with partner_knowledge={self.partner_knowledge}"

    def __post_init__(self):
        self.partner_knowledge = self.p1_policy.partner_knowledge
        self.timesteps = np.linspace(0, self.dynamics.N, self.dynamics.N + 1)
        self.set_empty_arrays()

        assert self.p1_policy.Q_partner == self.p2_policy.Q_self, "p1's partner Q must be equal to p2's self Q"
        assert self.p2_policy.Q_partner == self.p1_policy.Q_self, "p1's partner Q must be equal to p2's self Q"
        assert self.p1_policy.R_self_partner == self.p2_policy.R_partner_self, "p1's R_self_partner must be equal to p2's R_partner_self"
        assert self.p2_policy.R_self_partner == self.p1_policy.R_partner_self, "p2's R_self_partner must be equal to p1's R_partner_self"

        if "rfx" in self.dynamics.state_mapping:
            self.p1_control_state_idx = self.dynamics.state_mapping["rfx"]
            self.p1_control_state_idy = self.dynamics.state_mapping["rfy"]
            self.p2_control_state_idx = self.dynamics.state_mapping["lfx"]
            self.p2_control_state_idy = self.dynamics.state_mapping["lfy"]
        else:
            self.p1_control_state_idx = self.dynamics.state_mapping["rhvx"]
            self.p1_control_state_idy = self.dynamics.state_mapping["rhvy"]
            self.p2_control_state_idx = self.dynamics.state_mapping["lhvx"]
            self.p2_control_state_idy = self.dynamics.state_mapping["lhvy"]

    def set_empty_arrays(self):
        # * Define empty arrays
        self.x = (
            np.zeros((self.dynamics.N + 1, self.dynamics.A.shape[0], 1)) * np.nan
        )  # True states
        self.x1_pred = np.zeros_like(self.x) * np.nan
        self.x1_post = np.zeros_like(self.x) * np.nan
        self.x2_pred = np.zeros_like(self.x) * np.nan
        self.x2_post = np.zeros_like(self.x) * np.nan
        self.x_applied_force = np.zeros_like(self.x) * np.nan  # True states

        self.y1_obs = (
            np.zeros((self.dynamics.N + 1, self.dynamics.C1.shape[0], 1)) * np.nan
        )  # Observed state with noise
        self.y2_obs = (
            np.zeros((self.dynamics.N + 1, self.dynamics.C2.shape[0], 1)) * np.nan
        )  # Observed state with noise

        self.u1 = np.ones((self.dynamics.N, self.dynamics.B1.shape[1], 1)) * np.nan
        self.u2 = np.ones((self.dynamics.N, self.dynamics.B2.shape[1], 1)) * np.nan

        self.x_center_channel = (
            self.dynamics.x0
        )  # Set x_center_channel to start position

    def set_jump_perturbation_variables(self):
        self.jump_flag = True
        self.jump_back_flag = True
        self.jump_step = 10000  # Big number so doesn't trigger if statement until the jump is actually initiated
        self.jump_back_step = np.nan
        self.jump = np.zeros_like(self.x)  # Want this to be time varying
        self.linear_jump_steps = int(
            25 * (0.001 / self.dynamics.h)
        )  # 25ms linear jump out, like experiment
        self.jump_ids = [
            self.dynamics.state_mapping[key] for key in self.perturbation_states
        ]

    @property
    def process_noise1(self):
        ans = np.zeros_like(self.dynamics.x0)
        ans[self.p1_control_state_idx] = np.random.normal(0, self.process_noise_value)
        ans[self.p1_control_state_idy] = np.random.normal(0, self.process_noise_value)
        return ans

    @property
    def process_noise2(self):
        ans = np.zeros_like(self.dynamics.x0)
        ans[self.p2_control_state_idx] = np.random.normal(0, self.process_noise_value)
        ans[self.p2_control_state_idy] = np.random.normal(0, self.process_noise_value)
        return ans

    def set_perturbation(self, t, probe_trial):
        # Initiate jump based on ccy distance for reaching, rty/lty distance for posture
        if self.movement_type == "reaching":
            initiate_jump = (
                self.x[t, self.dynamics.state_mapping["ccy"]]
                >= self.perturbation_onset_distance
                and self.jump_flag
            )
        else:
            initiate_jump = (
                self.x[t, self.dynamics.state_mapping["rty"]]
                <= self.perturbation_onset_distance
                and self.jump_flag
            )

        if initiate_jump:
            for jump_state_key, pert_distance in zip(
                self.perturbation_states, self.perturbation_distances
            ):
                jump_idx = self.dynamics.state_mapping[jump_state_key]
                self.jump[t : t + self.linear_jump_steps, jump_idx, 0] = np.repeat(
                    pert_distance / self.linear_jump_steps, self.linear_jump_steps
                )
            self.jump_flag = False  # set to false so we don't jump twice
            self.jump_step = t
        elif probe_trial:
            # Jump back
            if t >= self.jump_step + self.probe_duration and self.jump_back_flag:
                for jump_idx, pert_distance in zip(
                    self.jump_ids, self.perturbation_distances
                ):
                    self.jump[t : t + self.linear_jump_steps, jump_idx, 0] = np.repeat(
                        -pert_distance / self.linear_jump_steps, self.linear_jump_steps
                    )
                self.jump_back_step = t
                self.jump_back_flag = False

    def run_simulation(self, probe_trial):
        assert self.dynamics.isDiscretized
        self.set_jump_perturbation_variables()

        A = self.dynamics.A if not probe_trial else self.dynamics.A_probe
        B1 = self.dynamics.B1
        B2 = self.dynamics.B2
        C1 = self.dynamics.C1
        C2 = self.dynamics.C2
        x0 = self.dynamics.x0
        N = self.dynamics.N

        # * Initialize arrays
        self.x[0, :, :] = x0
        self.x_applied_force[0, :, :] = x0

        self.y1_obs[0, :, :] = C1 @ x0
        self.x1_post[0, :, :] = (
            x0  # initial condition, can't use prior because it's the first one
        )

        self.y2_obs[0, :, :] = C2 @ x0
        self.x2_post[0, :, :] = (
            x0  # initial condition, can't use prior because it's the first one
        )

        for t in range(0, N):
            # * Get optimal control signal
            self.u1[t] = -self.p1_policy.F[t] @ (self.x1_post[t])
            self.u2[t] = -self.p2_policy.F[t] @ (self.x2_post[t])

            # * Set self.jump
            self.set_perturbation(t, probe_trial=probe_trial)

            # * Update the state
            self.x[t + 1] = (
                A @ self.x[t]
                + B1 @ self.u1[t]
                + B2 @ self.u2[t]
                + self.process_noise1
                + self.process_noise2
                + self.jump[t]
            )

            self.y1_obs[t + 1], self.x1_pred[t + 1], self.x1_post[t + 1] = (
                self.p1_state_estimator.get_estimate(
                    t=t,
                    measurement=self.x[t + 1],
                    prior=self.x1_post[t],
                    u_self=self.u1[t],
                    u_partner=self.u2[t],
                    partner_knowledge=self.p1_policy.partner_knowledge,
                )
            )

            self.y2_obs[t + 1], self.x2_pred[t + 1], self.x2_post[t + 1] = (
                self.p2_state_estimator.get_estimate(
                    t=t,
                    measurement=self.x[t + 1],
                    prior=self.x2_post[t],
                    u_self=self.u2[t],
                    u_partner=self.u1[t],
                    partner_knowledge=self.p2_policy.partner_knowledge,
                )
            )

    @property
    def p1_applied_force(self):
        return self.x[:-1, self.p1_control_state_idx].squeeze()

    @property
    def p2_applied_force(self):
        return self.x[:-1, self.p2_control_state_idx].squeeze()

    @property
    def jump_time(self):
        return self.jump_step / (0.001 / self.dynamics.h)

    @property
    def jump_back_time(self):
        return self.jump_back_step / (0.001 / self.dynamics.h)


@dataclass
class VisualizeModel:
    model_simulation: ModelSimulator | DualModelSimulator

