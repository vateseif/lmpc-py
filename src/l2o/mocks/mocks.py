from llm import Plan

mock_plan = Plan(tasks=[
  "move the gripper to cube_4 and avoid any collision with every cube",
  "close gripper",
  "move the gripper 0.1m on top of cube_2 and avoid colliding with cube_2, cube_3 and cube_1",
  "open gripper",
  "move gripper to cube_3 and avoid collisions with every cube",
  "close gripper",
  "move gripper 0.1m on top of cube_4 and avoid collisions with every cube apart from cube_3",
  "open gripper",
  "move gripper to cube_1 and avoid collisions with every cube",
  "close gripper",
  "move gripper 0.1m above cube_3 and avoid collisions with every cube apart from cube_1",
  "open gripper"
])