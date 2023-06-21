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

SYSTEM_PROMPT_SIMPLE = """
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

class GPTTuner:
  def __init__(self, function_call):
    self.messages = [
                {"role": "system", "content": SYSTEM_PROMPT_SIMPLE},
            ]
    
    self.functions = {"update": function_call}


  def next_action(self, iteration, feedback_message):
    
    self.messages.append({"role": "user", "content": f'Episode {iteration}: {feedback_message}'})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages = self.messages,
        functions=FUNCTIONS,
        max_tokens=256,
    )
    
    try:
      fn = completion.choices[0].message["function_call"].name
      args = completion.choices[0].message["function_call"].arguments
      message_string = fn + args
    except:
      print("Retrying...")
      return self.next_action()
       
    #print message
    print('\033[91m' + message_string + '\033[0m')
    
    # apply message
    result = self.functions[fn](*list(ast.literal_eval(args).values()))
    
    return result