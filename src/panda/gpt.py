import os
import openai

from controller import Controller


openai.api_key = open(os.path.dirname(__file__) + '/keys/gpt4.key', 'r').readline().rstrip()


SYSTEM_PROMPT = open(os.path.dirname(__file__) + '/prompts/system.prompt', 'r').read()


class GPTAgent:
  def __init__(self, robot: Controller, sim):

    self.robot = robot
    self.sim = sim

    self.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]
    

  def next_action(self, feedback_message=None):
    if feedback_message is not None:
      self.messages.append({"role": "user", "content": feedback_message})
    completion = openai.ChatCompletion.create(
        model="gpt-4",
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

    robot_function = message_string.split("robot.")[1].split("(")[0]
    function_argument = message_string.split(f"robot.{robot_function}(")[1].split(")")[0]

    if len(function_argument)==0:
      self.robot.functions[robot_function]()
    else:
      self.robot.functions[robot_function](self.sim.functions[function_argument]())
    
    '''
    # parse arguments
    parametername = message_string.split("update(")[1].split(",")[0]
    parametervalue = float(message_string.split(f"update({parametername}, ")[1].split(")")[0])
    parametername = parametername.replace("'", "").replace('"', '')
    # apply function
    self.functions['update'](parametername, parametervalue)
    '''

'''
Your goal is that of picking a cube on the table and moving it to its target position. The position of the cube is `x_cube` while the position you should move the cube to is `x_cube_target`.
'''