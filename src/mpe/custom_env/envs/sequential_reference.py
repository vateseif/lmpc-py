import numpy as np
import seaborn as sns

from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World, alphabet
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env


class Scenario(BaseScenario):
    def make_world(self, num_agents, num_landmarks):
        # num_pairs: number of speaker-listener pairs
        num_pairs = int(num_agents/2)
        world = World()
        # set any world properties first
        world.dim_c = len(alphabet)
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.i = i
            agent.name = f'speaker_{i}' if i < num_pairs else f'listener_{i - num_pairs}'
            agent.collide = False
            agent.size = 0.075
            agent.movable = i>=num_pairs
            agent.silent = i>=num_pairs
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # add all possible colors of agents and landmarks
        world.colors = sns.color_palette(n_colors=num_landmarks)
        world.colors = [list(ci) for ci in world.colors]
        return world

    def reset_world(self, world, np_random):
        num_pairs = int(len(world.agents)/2) # number of speaker-listener pairs
        # set random initial states
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.color = np.array(world.colors[i])
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # assign goals to agents
        for i, agent in enumerate(world.agents):
            if agent.name.startswith('speaker'):
                agent.goal_a = world.agents[i+num_pairs]
                agent.goal_b = np.random.choice(world.landmarks)
                # color of listener = color of target landmark
                agent.goal_a.color = agent.goal_b.color + np.array([0.25, 0.25, 0.25])
                # color of speaker is grey
                agent.color = np.array([0.25, 0.25, 0.25])
            else:     
                agent.goal_a = None
                agent.goal_b = None
                
    def benchmark_data(self, agent, world: World):
        # returns data for benchmarking purposes
        return self.reward(agent, world)

    def reward(self, agent, world):
        # squared distance from listener to landmark        
        num_pairs = int(len(world.agents)/2)
        a = agent if agent.i<num_pairs else world.agents[agent.i-num_pairs]
        dist2 = 0.0
        for landmark in world.landmarks:
          if not landmark.draw_identity: continue
          dist2 = np.sum(np.square(a.goal_a.state.p_pos - landmark.state.p_pos))
          return -dist2
        return -dist2

    def observation(self, agent, world):
        num_pairs = int(len(world.agents)/2) # number of speaker-listener pairs
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for i, entity in enumerate(world.landmarks):
            delta_pos = entity.state.p_pos - agent.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            entity_pos.append(delta_pos)
            # minimum allowable distance
            dist_min = entity.size + agent.size
            # if landmark wasn't already collected and if distance
            # is smaller than the sum of radii then don't draw landmark anymore
            if entity.draw_entity and dist < dist_min:
              world.landmarks[i].draw_entity = False

        # communication of speaker to listener
        comm = [world.agents[agent.i-num_pairs].state.c]

        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])
        # listener
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)



class raw_env(SimpleEnv):
    def __init__(self, num_agents=None, num_landmarks=None, max_cycles=25, continuous_actions=False):
        scenario = Scenario()
        if num_agents==None: num_agents = 2
        if num_landmarks==None: num_landmarks = 3 
        world = scenario.make_world(num_agents, num_landmarks)
        super().__init__(scenario, world, max_cycles, continuous_actions)
        self.metadata["name"] = "simple_speaker_listener_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

