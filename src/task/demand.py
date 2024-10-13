
import enum

import numpy as np

from ..network import Cluster


class Circuit(enum.Enum):
    pass


class Demand:
    def __init__(self, ) -> None:

        self.qubits = []
        self.demands = {}


class RandomDemand(Demand):
    def __init__(self, 
            qubit_num: int=10,
            demand_frac: float=0.5,
            demand_range: tuple=(1, 11),
            ) -> None:
        self.qubits = range(qubit_num)
        self.demands = self.get_demand(demand_frac, demand_range)

        
    def get_demand(self, frac: float, demand_range: tuple) -> dict:
        pairs = []
        for src in self.qubits:
            for dst in self.qubits:
                if src != dst and (dst, src) not in pairs:
                    pairs.append((src, dst))
        indices = range(len(pairs))
        indices = np.random.choice(indices, int(frac*len(pairs)), replace=False)

        demands = {}
        for i in indices:
            pair = pairs[i]
            demands[pair] = np.random.randint(*demand_range)

        return demands


if __name__ == '__main__':
    demand = RandomDemand(10, 0.5, (1, 11))
    print(demand.demands)
