import numpy as np

class PID:
  def __init__(self, dim=4) -> None:

    self.name = 'pid'
  
    # number of states
    self.dim = dim

    # desired state (= 0 because regulator by default)
    self.desired_state = np.zeros((dim,))
    
    # pid parameters
    self.P = 0.0    # target = 50.0
    self.I = 0.0    # target = 0.005
    self.D = 0.0    # target = 1.0
    self.parameters = {'P':self.P, 'I':self.I, 'D':self.D}

    self.dt = 0.02 # = tau in cartpole

    # weight on tracking error for each state (=1 they are valued equally)
    self.mask = np.array([0.1, 0.1, 1.0, 0.1])

    self.integral = 0.    # accumulated error
    self.derivative = 0.  # derivative error
    self.prev_error = 0.  # previous error

  def reset(self):
    self.integral = 0.    # accumulated error
    self.derivative = 0.  # derivative error
    self.prev_error = 0.  # previous error

  def update(self, parameter_name:str, parameter_value:float):
    assert parameter_name in list(self.parameters.keys())
    # update parameters value
    setattr(self, parameter_name, parameter_value)
    self.parameters[parameter_name] = parameter_value

  def act(self, x: np.ndarray):

    error = x - self.desired_state
    self.integral += error
    self.derivative = (error - self.prev_error) / self.dt
    self.prev_error = error

    action = np.dot(self.P * error + self.I * self.integral + self.D * self.derivative, self.mask)
    action = min(action, 10)
    action = max(action, -10)
    return action


  @staticmethod
  def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
