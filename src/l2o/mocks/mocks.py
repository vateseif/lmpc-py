from llm import Plan

mock_plan = Plan(tasks=[
  "go above cube_1",
  "go to cube_1",
  "close gripper",
  "go above cube_2 without colliding with it",
  "open gripper",
  "go -0.2m of the y axis of "
])