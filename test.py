import time
from IPython.display import clear_output



from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from Algorithms import *

DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

MAPS = {
    "2x2": ["SD", 
            "DG"],
    "4x4": ["SFFF",
            "FDFF",
            "FFFD",
            "FFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFTAL",
        "TFFHFFTF",
        "FFFFFHTF",
        "FAFHFFFF",
        "FHHFFFHF",
        "DFTFHDTL",
        "FLFHFFFG",
    ],
    "20x20": [
        'SFFLHFHTALHLFATAHTHT',
        'HFTTLLAHFTAFAAHHTLFH',
        'HHTFFFHAFFFFAFFTHHHT',
        'TTAFHTFHTHHLAHHAALLF',
        'HLALHFFTHAHHAFFLFHTF',
        'AFTAFTFLFTTTFTLLTHDF',
        'LFHFFAAHFLHAHHFHFALA',
        'AFTFFLTFLFTAFFLTFAHH',
        'HTTLFTHLTFAFFLAFHFTF',
        'LLALFHFAHFAALHFTFHTF',
        'LFFFAAFLFFFFHFLFFAFH',
        'THHTTFAFLATFATFTHLLL',
        'HHHAFFFATLLALFAHTHLL',
        'HLFFFFHFFLAAFTFFDAFH',
        'HTLFTHFFLTHLHHLHFTFH',
        'AFTTLHLFFLHTFFAHLAFT',
        'HAATLHFFFHHHHAFFFHLH',
        'FHFLLLFHLFFLFTFFHAFL',
        'LHTFLTLTFATFAFAFHAAF',
        'FTFFFFFLFTHFTFLTLHFG']
}

env = DragonBallEnv(MAPS["8x8"])
state = env.reset()
print('Initial state:', state)
print('Goal states:', env.goals)

def print_solution(actions,env: DragonBallEnv) -> None:
    env.reset()
    total_cost = 0
    print(env.render())
    print(f"Timestep: {1}")
    print(f"State: {env.get_state()}")
    print(f"Action: {None}")
    print(f"Cost: {0}")
    time.sleep(1)

    for i, action in enumerate(actions):
      state, cost, terminated = env.step(action)
      total_cost += cost
      clear_output(wait=True)

      print(env.render())
      print(f"Timestep: {i + 2}")
      print(f"State: {state}")
      print(f"Action: {action}")
      print(f"Cost: {cost}")
      print(f"Total cost: {total_cost}")
      
      time.sleep(1)

      if terminated is True:
        break
      
def test_bfs():
    BFS_agent = BFSAgent()
    actions, total_cost, expanded = BFS_agent.search(env)
    print(f"Total_cost: {total_cost}")
    print(f"Expanded: {expanded}")
    print(f"Actions: {actions}")
    assert total_cost == 119.0, "Error in total cost returned"
    return actions, env

def test_astar():
    WA_agent = WeightedAStarAgent()
    actions, total_cost, expanded = WA_agent.search(env, h_weight=0.5)
    print(f"Total_cost: {total_cost}")
    print(f"Expanded: {expanded}")
    print(f"Actions: {actions}")

    return actions, env

def test_astar_epsilon(epsilon=100):
    print("epsilon: ", epsilon)
    AStarEpsilon_agent = AStarEpsilonAgent()
    actions, total_cost, expanded = AStarEpsilon_agent.search(env, epsilon=epsilon)
    print(f"Total_cost: {total_cost}")
    print(f"Expanded: {expanded}")
    print(f"Actions: {actions}")
    return actions, env

if __name__ == "__main__":
    # actions, env = test_bfs()
    actions, env = test_astar()
    actions, env = test_astar_epsilon(epsilon=100)
   # print_solution(actions, env)
    