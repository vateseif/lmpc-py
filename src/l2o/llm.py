import os
import openai
import numpy as np

from core import AbstractLLM
from config.config import OptimizationDesignerConfig, TaskPlannerConfig

from typing import List
from pydantic import BaseModel, Field
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser


# GPT4 api key
os.environ["OPENAI_API_KEY"] = open(os.path.dirname(__file__) + '/keys/gpt4.key', 'r').readline().rstrip()


class Plan(BaseModel):
  tasks: List[str] = Field(description="list of all tasks that the robot has to carry out")
  
class Optimization(BaseModel):
  objective: str = Field(description="objective function to be applied to MPC")
  #constraints: List[str] = Field(description="constraints to tbe applied to MPC written in cvxpy")


class TaskPlanner(AbstractLLM):
  def __init__(self, cfg=TaskPlannerConfig()) -> None:
    super().__init__(cfg)
    # init parser
    self.parser = PydanticOutputParser(pydantic_object=Plan)
    # init GPT model
    self.model = ChatOpenAI(model_name=self.cfg.model_name, temperature=self.cfg.temperature)
    # init system prompt
    system_message_prompt = SystemMessagePromptTemplate.from_template(self.cfg.prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
    self.prompt = chat_prompt.format_prompt(format_instructions=self.parser.get_format_instructions()).to_messages()

  def run(self) -> Plan:
    output = self.model(self.prompt)
    plan = self.parser.parse(output.content)
    print(plan)
    return plan 


class OptimizationDesigner(AbstractLLM):
  def __init__(self, cfg=OptimizationDesignerConfig()) -> None:
    super().__init__(cfg)
    # init parser
    self.parser = PydanticOutputParser(pydantic_object=Optimization)
    # init prompt
    system_prompt = SystemMessagePromptTemplate.from_template(self.cfg.prompt)
    self.messages = [system_prompt.format(format_instructions=self.parser.get_format_instructions())]    
    # init model
    self.model = ChatOpenAI(model_name=self.cfg.model_name, temperature=self.cfg.temperature)


  def run(self, user_message:str) -> str:
    self.messages.append(HumanMessage(content=user_message))
    model_message = self.model(self.messages)
    self.messages.append(model_message)
    print(f"\033[91m {model_message.content} \033[0m")
    return self.parser.parse(model_message.content)



OptimizationDesignerOptions = {
  "objective": OptimizationDesigner
}