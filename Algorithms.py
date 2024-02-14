import numpy as np

from abc import ABC, abstractmethod 
import time

from DragonBallEnv import DragonBallEnv
from IPython.display import clear_output
from typing import List, Tuple
import heapdict

class Agent(ABC):
    def __init__(self) -> None:
        self.env = None

    def print_solution(self, actions) -> None:
        self.env.reset()
        total_cost = 0
        print(self.env.render())
        print(f"Timestep: {1}")
        print(f"State: {self.env.get_state()}")
        print(f"Action: {None}")
        print(f"Cost: {0}")
        time.sleep(1)

        for i, action in enumerate(actions):
            state, cost, terminated = self.env.step(action)
            total_cost += cost
            clear_output(wait=True)

            print(self.env.render())
            print(f"Timestep: {i + 2}")
            print(f"State: {state}")
            print(f"Action: {action}")
            print(f"Cost: {cost}")
            print(f"Total cost: {total_cost}")
            
            time.sleep(1) 

            if terminated is True:
                break
      
    @abstractmethod
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        pass

class Node:
    def __init__(self, state: int, parent , cost: float, action: int, heuristic: float = None) -> None:
        self.state = state
        self.parent = parent
        self.cost = cost
        self.action = action
        # self.heuristic = heuristic

    # def __lt__(self, other) -> bool:
    #     return self.cost + self.heuristic < other.cost + other.heuristic
    #     return self.cost < other.cost

    def __hash__(self) -> int:
        return hash(self.state)

    def __str__(self) -> str:
        return f"State: {self.state}, Parent: {self.parent}, Action: {self.action}, Cost: {self.cost}, Heuristic: {self.heuristic}"
class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def _get_path_total_cost(self, node: Node) -> float:
        total_cost = 0
        while node is not None:
            total_cost += node.cost
            node = node.parent
        return total_cost

    def _get_path_actions(self, node: Node) -> List[int]:
        actions = []
        if Node is None:
            return []
        
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return actions[::-1]   
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        # print("d2 : ", self.env.d2)
        # print("d1: ", self.env.d1)
        # Graph bfs
        state = self.env.get_initial_state()
        node = Node(state,  None, 0, None) # Make a node.
        if (self.env.is_final_state(state)):
            return [], 0, 0
        open = [node]
        closed = set()
        expanded_nodes = 0
        while len(open) > 0:
            node = open.pop(0)
            if node.state[0] == self.env.d1[0]:
                node.state = node.state[0], True, node.state[2]
                # print('found first  dragon ball')
                # self.print_solution(self._get_path_actions(node))

            if node.state[0] == self.env.d2[0]:
                node.state = node.state[0], node.state[1], True
                # print('found second dragon ball')
                # self.print_solution(self._get_path_actions(node))
            closed.add(node.state)
            expanded_nodes += 1
            for action, (next_state, cost, terminated) in env.succ(node.state).items():
                next_state = next_state[0], node.state[1], node.state[2]
                child = Node(next_state, node, cost, action)

                if terminated is True and self.env.is_final_state(child.state) is False:
                    # state is a hole
                    continue
                if child.state not in closed and child not in open:
                    if self.env.is_final_state(child.state):
                        print('Found a solution !')
                        return self._get_path_actions(child), self._get_path_total_cost(child), expanded_nodes
                    open.append(child)
        print('couldn\'t find a solution')
                    

                

        


class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError