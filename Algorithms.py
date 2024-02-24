import numpy as np

from abc import ABC, abstractmethod 
import time

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class Node:
    def __init__(self, state: int, parent ,cost: float, action: int, heuristic: float = None, terminated: bool = False) -> None:
        self.state = state
        self.parent = parent
        self.cost = cost
        self.action = action
        self.heuristic = heuristic
        self.terminated = terminated
        
    def __eq__(self, other) -> bool:
        return other.state == self.state
    
    def __hash__(self):
        return hash(self.state)
    def __str__(self) -> str:
        return f"State: {self.state}, Parent: {self.parent}, Action: {self.action}, Cost: {self.cost}, Heuristic: {self.heuristic}"
    
class Agent(ABC):
    def __init__(self) -> None:
        self.env = None

    # Helper functions that get the total cost and the actions of the node based on the path to that node.
    def _get_path_total_cost(self, node: Node) -> float:
        # Only for BFS
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
    
    def manhattan_distance(self, first_state, second_state):
        first_row, first_col = self.env.to_row_col(first_state)
        second_row, second_col = self.env.to_row_col(second_state)
        return abs(first_row - second_row) + abs(first_col - second_col)

    def hmsap(self, state, old_state=None) -> int:
        if old_state is not None:
            collected_d1, collected_d2 = old_state[1], old_state[2]
        else:
            collected_d1, collected_d2 = False, False   
        distant_from_d1 = self.manhattan_distance(state, self.env.d1) if not collected_d1 else np.inf
        distant_from_d2 = self.manhattan_distance(state, self.env.d2) if not collected_d2 else np.inf
        
        distant_from_goals = [self.manhattan_distance(state, goal) for goal in self.env.get_goal_states()]
        return min(distant_from_goals + [distant_from_d1, distant_from_d2])
    
    def f(self, node: Node, h_weight) -> float:
        return ((1 - h_weight) * node.cost + h_weight * node.heuristic, node.state)
    
    @abstractmethod
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        pass

class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
       
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        # print("d2 : ", self.env.d2)
        # print("d1: ", self.env.d1)
        # Graph bfs
        state = self.env.get_initial_state()
        node = Node(state,  None, 0, None) # Make a node.
        if self.env.is_final_state(state):
            return [], 0, 0
        open = [node]
        closed = set()
        expanded_nodes = 0
        while len(open) > 0:
            node = open.pop(0)
            closed.add(node.state)
            expanded_nodes += 1
            for action, (next_state, cost, terminated) in env.succ(node.state).items():
                if node.terminated is True:
                    # Can't continue moving (a hole or "G")
                    break

                child = Node(next_state, node, cost, action, terminated=terminated)
                # Set child dragon ball's to true in 2 cases:
                # a. Parent found the dragon ball; b. Child location is the dragon ball location.
                child.state = child.state[0], \
                              node.state[1] or child.state[0] == self.env.d1[0], \
                              node.state[2] or child.state[0] == self.env.d2[0]

                if child.state not in closed and child not in open:
                    if self.env.is_final_state(child.state):
                        return self._get_path_actions(child), self._get_path_total_cost(child), expanded_nodes
                    open.append(child)
                    
class WeightedAStarAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
    
    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()
        node = Node(state,  parent=None, cost=0, action=None, heuristic=self.hmsap(state)) # Make a node.

        open = heapdict.heapdict({node: self.f(node, h_weight)})
        closed = {} # key: state, value: node
        
        expanded_nodes = 0
        while len(open) > 0:
            node, node_fval = open.popitem()
            closed[node.state] = node
            if self.env.is_final_state(node.state):
                return self._get_path_actions(node), node.cost, expanded_nodes
            
            expanded_nodes += 1
            for action, (next_state, edge_cost, terminated) in env.succ(node.state).items():
                if node.terminated is True:
                    # Can't continue moving (a hole or "G")
                    break
                cost = node.cost + edge_cost # Cost==gvalue
                child = Node(next_state, node, cost, action, terminated=terminated)
                # Set child dragon ball's to true in 2 cases:
                # a. Parent found the dragon ball; b. Child location is the dragon ball location.
                child.state = child.state[0], \
                              node.state[1] or child.state[0] == self.env.d1[0], \
                              node.state[2] or child.state[0] == self.env.d2[0]  
                        
                child.heuristic=self.hmsap(child.state, old_state=node.state)
                fval = self.f(child, h_weight)

 
                if child.state not in closed and child not in open:
                    open[child] = fval
                elif child in open:
                    old_fval = open[child]
                    if fval < old_fval:
                        open.pop(child) # Drop the old node.
                        open[child] = fval  # Append the new node with the new fval.
                else: 
                    # Child in close
                    old_child = closed[child.state]
                    fval, old_fval = self.f(child, h_weight), self.f(old_child, h_weight)
                    if fval < old_fval:
                        open[child] = fval
                        closed.pop(child.state)

class AStarEpsilonAgent(Agent):
    def __init__(self) -> None:
        self.env = None

    def focal_min(self, open, epsilon, h_focal):
        _, min_fval = open.peekitem()
        min_fval = min_fval[0] # Consier only the f-value, ignore state.
        focal_set = []
        for node, node_fval in open.items():
            node_fval = node_fval[0]
            if node_fval <= (1 + epsilon) * min_fval:
                focal_set.append((node, h_focal(node)))
        
        return min(focal_set, key=lambda x: x[1])[0]


    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()
        node = Node(state,  parent=None, cost=0, action=None, heuristic=self.hmsap(state)) # Make a node.

        open = heapdict.heapdict({node: self.f(node, h_weight=0.5)})
        closed = {} # key: state, value: node
        
        expanded_nodes = 0
        while len(open) > 0:
            node = self.focal_min(open, epsilon, h_focal=lambda node: node.cost)
            open.pop(node)
            closed[node.state] = node
            if self.env.is_final_state(node.state):
                return self._get_path_actions(node), node.cost, expanded_nodes
            
            expanded_nodes += 1
            for action, (next_state, edge_cost, terminated) in env.succ(node.state).items():
                if node.terminated is True:
                    # Can't continue moving (a hole or "G")
                    break
                cost = node.cost + edge_cost # Cost==gvalue
                child = Node(next_state, node, cost, action, terminated=terminated)
                # Set child dragon ball's to true in 2 cases:
                # a. Parent found the dragon ball; b. Child location is the dragon ball location.
                child.state = child.state[0], \
                              node.state[1] or child.state[0] == self.env.d1[0], \
                              node.state[2] or child.state[0] == self.env.d2[0]
                         
                child.heuristic=self.hmsap(child.state, old_state=node.state)
                fval = self.f(child, h_weight=0.5)

                if child.state not in closed and child not in open:
                    open[child] = fval
                elif child in open:
                    old_fval = open[child]
                    if fval < old_fval:
                        open.pop(child) # Drop the old node.
                        open[child] = fval  # Append the new node with the new fval.
                else: 
                    # Child in close
                    old_child = closed[child.state]
                    fval, old_fval = self.f(child, h_weight=0.5), self.f(old_child, h_weight=0.5)
                    if fval < old_fval:
                        open[child] = fval
                        closed.pop(child.state)
