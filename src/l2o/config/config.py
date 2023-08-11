from core import AbstractControllerConfig, AbstractLLMConfig
from prompts.stack import TASK_PLANNER_PROMPT, OPTIMIZATION_DESIGNER_PROMPT 



class TaskPlannerConfig(AbstractLLMConfig):
  prompt: str = TASK_PLANNER_PROMPT
  model_name: str = "gpt-3.5-turbo"
  temperature: float = 0.7


class OptimizationDesignerConfig(AbstractLLMConfig):
  prompt: str = OPTIMIZATION_DESIGNER_PROMPT
  model_name: str = "gpt-3.5-turbo"
  temperature: float = 0.7

class BaseControllerConfig(AbstractControllerConfig):
  nx: int = 3
  nu: int = 3 
  T: int = 15
  dt: float = 0.05
  lu: float = -0.5 # lower bound on u
  hu: float = 0.5  # higher bound on u
