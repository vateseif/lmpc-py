import lmpc
import numpy as np

# TODO lmpc is being imported twice. Fix all imports 

def getAgent():
  N = 1
  Ns = 4
  Na = 4
  T = 20
  dt = 0.1        # TODO get val from env
  tau = 0.25      # TODO get val from env
  sensitivity = 5 # TODO get val from env
  eps = 1e-3
  Q = np.zeros((Ns,Ns))
  R = np.eye(Na)

  A = np.array([[0, 0, 1, 0], 
                [0, 0, 0, 1], 
                [0, 0, -tau/dt, 0], 
                [0, 0, 0, -tau/dt]])
  B = np.array([[0, 0, 0, 0], 
                [0, 0, 0, 0], 
                [1, -1, 0, 0], 
                [0, 0, 1, -1]]) 

  Ad = np.eye(Ns) + A*dt
  Bd = B*dt*sensitivity

  sys = lmpc.DistributedLTI(N, Ns, Na)
  sys.loadAB(Ad, Bd)

  controller = lmpc.LMPC(T)
  controller << sys
  # objective
  controller.addObjectiveFun(lmpc.objectives.QuadForm(Q, R))
  # box constraints control inputs
  controller.addConstraint(lmpc.BoundConstraint('u', 'upper', (1-eps)*np.ones((sys.Nu,1))))
  controller.addConstraint(lmpc.BoundConstraint('u', 'lower', eps * np.ones((sys.Nu,1))))

  xT = np.zeros((2,1))
  G = np.concatenate((np.eye(2), np.zeros((2,2))), axis=1)
  controller.addConstraint(lmpc.TerminalConstraint(xT, G))
  controller._setupSolver(np.zeros((4,1)))
  return controller

