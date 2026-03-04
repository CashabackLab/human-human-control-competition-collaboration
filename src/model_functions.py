import numpy as np
from copy import deepcopy

def augment_A_matrix(A, sensory_delay):
    out = np.block(
        [
            [
                A, np.zeros((A.shape[0], A.shape[1] * sensory_delay))
            ],
            [
                np.eye(A.shape[0] * sensory_delay, A.shape[1] * sensory_delay),
                np.zeros((A.shape[0] * sensory_delay, A.shape[1])),
            ],
        ]
    )
    return out


def augment_Q_matrix(Q, sensory_delay):
    if Q.ndim == 2:
        out = np.block(
            [
                [
                    Q, np.zeros((Q.shape[0], sensory_delay * Q.shape[1]))
                ],
                [
                    np.zeros((Q.shape[0] * sensory_delay, Q.shape[1])),
                    np.zeros((Q.shape[0] * sensory_delay, Q.shape[1] * sensory_delay)),
                ],
            ]
        )
    else:
        out = np.block(
            [
                [
                    Q, np.zeros((Q.shape[0], Q.shape[1], sensory_delay * Q.shape[2]))
                ],
                [
                    np.zeros((Q.shape[0], Q.shape[1] * sensory_delay, Q.shape[2])),
                    np.zeros((Q.shape[0], Q.shape[1] * sensory_delay, Q.shape[2] * sensory_delay)),
                ],
            ]
        )

    return out


def augment_B_matrix(B, sensory_delay):
    out = np.block([[B], [np.zeros((B.shape[0] * sensory_delay, B.shape[1]))]])
    return out