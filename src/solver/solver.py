
from abc import ABC, abstractmethod

import numpy as np
import gurobipy as gp

from ..network import Cluster, Topology
from ..task import Demand, RandomDemand



class Solver(ABC):
    def __init__(self, cluster: Cluster, demand: Demand) -> None:
        self.cluster = cluster
        self.demand = demand

        self.nodes = self.cluster.graph.nodes(data=False)
        self.edges = self.cluster.graph.edges(data=False)
        # all 2-tuple combinations of nodes
        self.pairs = []
        for i in self.nodes:
            for j in self.nodes:
                if i != j and (j, i) not in self.pairs:
                    self.pairs.append((i, j))

        self.model: 'gp.Model' = None

    @abstractmethod
    def build(self):
        pass

    def solve(self):
        self.model.optimize()

        if hasattr(self.model, 'ObjVal'):
            return self.model.ObjVal
        else:
            return None
