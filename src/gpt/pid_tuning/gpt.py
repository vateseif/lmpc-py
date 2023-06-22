import os
import ast
import openai

openai.api_key = open(os.path.dirname(__file__) + '/gpt3.key', 'r').readline().rstrip()

FUNCTIONS = [
      {
        "name": "update",
        "description": "Updates the parameters of the PID controller",
        "parameters":{
          "type": "object",
          "properties":{
            "parametername":{
              "type": "string",
              "description": "Name of the parameter to be changed. It can be only one of the following: 'P', 'I', 'D'",
            },
            "parametervalue":{
              "type": "number",
              "description": "New value to be assigned to parametername",
            }
          }
          
        },
        "required": ["parametername", "parametervalue"]
      }
  ]

SYSTEM_PROMPT_LONG = """
We are trying to control the cartpole environment in openai gym using a PID controller and we want you to help us tune the PID parameters.
Each time we run an episode of the simulation we will provide you feedback on the system behavior and you have to update the PID parameters based on your knowledge as a control engineer.

The cartpole has 4 states:
- cartpole position
- cartpole lateral velocity
- cartpole angle
- cartpole angular velocity

We define the system to have satisfying behavior when the pole stays upright and does not oscillate nor it goes unstable.
The PID parameters are initialized to the following values:
P = 1.0
I = 0.0
D = 0.0

You have access to the the PID parameters by using the following function:
```
update(parametername, parametervalue)
```

As an example, if the cartpole is oscillating too hard around the upright position you can reduce the P value as follows
```
update('P', 0.5)
```

Only you have access to the PID parameters, so when you change a prameter, it will remain as such until you change it again.
"""

SYSTEM_PROMPT_SIMPLE_PID = """
We are trying to control an inverted pendulum using a PID controller and we want you to help us tune the PID parameters.
Each time we run an episode of the simulation we will provide you feedback on the system behavior and you have to update the PID parameters based on your knowledge as a control engineer.
We define the system to have satisfying behavior when the pole stays upright and does not go unstable.
The PID parameters are initialized to the following values:
P = 0.0
I = 0.0
D = 0.0
You have access to the the PID parameters by using the following function:
```
update(parametername, parametervalue)
```
As an example, if the cartpole is oscillating too hard around the upright position you can reduce the P value as follows
```
update('P', 0.5)
```
You can update only 1 parameter at a time so call the update function only once for each time.
For the cartpole to be stable the right PID parameters may be far from where initialized, so don't hesitate to explore.
Only you have access to the PID parameters, so when you change a prameter, it will remain as such until you change it again.
Everytime an episode is run, we will provide feedback in the form:
```
Episode [EPISODE_NUM]: [FEEDBACK]
```
Use the feedback and the provided api function `update` to update the PID controller. Explain your reasoning every time.
"""

SYSTEM_PROMPT_SIMPLE_LQR = """
We are trying to control an inverted pendulum using an LQR controller and we want you to help us tune the LQR parameters.
Each time we run an episode of the simulation we will provide you feedback on the system behavior and you have to update the LQR parameters based on your knowledge as a control engineer.
We define the system to have satisfying behavior when the pole stays upright and does not go unstable.
The LQR parameters are that you can change are the gains for each state.

The inverted pendulum states are: 
- x: the position of the cart
- dx: the velocity of the cart
- theta: the angle of the pendulum on the cart. theta=0 indicates the pendulum is in the upright position.
- dtheta: the angular velocity of the pendulum on the cart.

The state gains are the parameters you have access to and they are initialized to the following values:
- x = 1.0
- dx = 1.0
- theta = 1.0
- dtheta = 1.0

You have access to the the LQR gain parameters by using the following function:
```
update(parametername, parametervalue)
```
As an example, if the cartpole is oscillating too hard around the upright position you can increase the gain of x state as follows:
```
update('x', 1.0)
```
You can update only 1 parameter at a time so call the update function only once for each time.
Only you have access to the LQR parameters, so when you change a prameter, it will remain as such until you change it again.
Everytime an episode is run, we will provide feedback in the form:
```
Episode [EPISODE_NUM]: [FEEDBACK]
```
Use the feedback and the provided api function `update` to update the LQR controller. Explain your reasoning every time.
"""

class GPTTuner:
  def __init__(self, ctrl):

    # prompt depends on type of controller
    self.system_prompts = {
      'pid': SYSTEM_PROMPT_SIMPLE_PID,
      'lqr': SYSTEM_PROMPT_SIMPLE_LQR
    }

    self.messages = [
                {"role": "system", "content": self.system_prompts[ctrl.name]},
            ]
    
    self.functions = {"update": ctrl.update}


  def next_action(self, iteration=None, feedback_message=None):
    if feedback_message is not None:
      self.messages.append({"role": "user", "content": f'Episode {iteration}: {feedback_message}'})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = self.messages,
        #functions=FUNCTIONS,
        #max_tokens=256,
    )
    
    try:
      #fn = completion.choices[0].message["function_call"].name
      #args = completion.choices[0].message["function_call"].arguments
      #message_string = fn + args
      message_string = completion.choices[0].message.content

    except:
      print("Retrying...")
      return self.next_action()

    # store gpt reply
    self.messages.append({"role": "assistant", "content": message_string})
       
    # apply
    self.apply_action(message_string)
    
    
    
  def apply_action(self, message_string):
    # print
    print(f"\033[91m {message_string} \033[0m")
    # parse arguments
    parametername = message_string.split("update(")[1].split(",")[0]
    parametervalue = float(message_string.split(f"update({parametername}, ")[1].split(")")[0])
    parametername = parametername.replace("'", "").replace('"', '')
    # apply function
    self.functions['update'](parametername, parametervalue)