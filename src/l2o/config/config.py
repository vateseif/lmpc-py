from core import AbstractControllerConfig, AbstractLLMConfig, AbstractRobotConfig
from prompts.stack import TASK_PLANNER_PROMPT, OPTIMIZATION_DESIGNER_PROMPT 


class PlanLLMConfig(AbstractLLMConfig):
  prompt: str = TASK_PLANNER_PROMPT
  parsing: str = "plan"
  model_name: str = "gpt-4"
  temperature: float = 0.7

class ObjectiveLLMConfig(AbstractLLMConfig):
  prompt: str = OPTIMIZATION_DESIGNER_PROMPT
  parsing: str = "objective"
  model_name: str = "gpt-4"
  temperature: float = 0.7

class OptimizationLLMConfig(AbstractLLMConfig):
  prompt: str = OPTIMIZATION_DESIGNER_PROMPT
  parsing: str = "optimization"
  model_name: str = "gpt-3.5-turbo"
  temperature: float = 0.7

class BaseControllerConfig(AbstractControllerConfig):
  nx: int = 3
  nu: int = 3 
  T: int = 15
  dt: float = 0.05
  lu: float = -0.5 # lower bound on u
  hu: float = 0.5  # higher bound on u


class BaseRobotConfig(AbstractRobotConfig):
  name: str = "objective"
  od_type: str = "objective"
  controller_type: str = "objective"


BaseLLMConfigs = {
  "plan": PlanLLMConfig,
  "objective": ObjectiveLLMConfig,
  "optimization": OptimizationLLMConfig
}
