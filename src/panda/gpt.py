import os
import openai
import numpy as np

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
    

  def next_action(self, feedback_message=None, role="user"):
    if feedback_message is not None:
      self.messages.append({"role": role, "content": feedback_message})
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
    function_args = message_string.split(f"robot.{robot_function}(")[1].split(")")[0]
    
    if len(function_args)==0:
      self.robot.functions[robot_function]()
    else:
      # arguments of function
      xd_arg = function_args.split(",")[0]
      offset_arg = function_args.split("[")[1].split("]")[0]
      # retriev values from string arguments
      xd = self.sim.functions[xd_arg]()
      offset = np.fromstring(offset_arg,sep=',')
      # apply
      self.robot.functions[robot_function](xd, offset)
