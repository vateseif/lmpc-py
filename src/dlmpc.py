"""
Reimplementing in python Fig5_Terminal_constraints/scenario1_tc.m
"""
import numpy as np

# Network params
NETWORK = 4
LOCALITY = 3

# State dimensions
N = NETWORK   # number of pendulums
NX = 2*N      # each pendulum has 2 states (theta, dtheta)
NU = N        # each pendulum has only dtheta actuated

# System params
m = 1         # [kg] mass of the pendulum
k = 1         # [N/m] spring constant
d = 3         # [Ns/m] damping constant
g = 10        # [m/s^2] gravity
l = 1         # [m] length of pendulum

# Scenario parameters
d = LOCALITY
x0 = 0.5 * np.ones((NX, 1))
T = 10


# Construct dynamic system matrices: dx = Ax + Bu
block_off_diag = np.array([[0,0], [k*l/m, d/m/l]])
block_diag_extra = np.array([[0, 1], [-g-k*l/m, -d/(m*l)]])
block_diag = np.array([[0, 1], [-g-2*k*l/m, -2*d/(m*l)]])

Ac = np.zeros((NX, NX))
Bc = np.zeros((NX, NU))
for j, i in enumerate(range(0, NX, 2)):
  Ac[i:i+2,i:i+2] = block_diag
  if j!=N-1:
    Ac[i:i+2,i+2:i+4] = block_off_diag
  if j!=0:
    Ac[i:i+2,i-2:i] = block_off_diag

  Bc[i:i+2, j] = np.array([0, 1])

# Discretize system
TS = .1       # [s] sampling time

A = np.eye(NX) + TS * Ac
B = TS * Bc

# cost matrices
Q = np.eye(NX)
S = np.eye(NU)

# feasibility constraints
I = np.eye(NX*T)

Z = np.eye(NX*(T-1))
Z = np.concatenate((Z, np.zeros((NX*(T-1), NX))), axis=1)
Z = np.concatenate((np.zeros((NX, NX*T)), Z), axis=0)

E1 = np.concatenate((np.eye(NX), np.zeros((NX*T, NX)))) 

Aa = np.kron(np.eye(T, dtype=int), A)   # blkdiag(A, ..., B)
Bb = np.kron(np.eye(T, dtype=int), B)   # blkdiag(B, ..., B)


      