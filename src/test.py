
import time
import copy
import numpy as np

from .network import Cluster, Topology
from .task import RandomDemand
from .solver import CMF, CMP

if __name__ == "__main__":

    seed = 0
    np.random.seed(seed)
    
    node_mem = 2
    qubit_num = 8
    gate_range = (8, 16)
    cluster = Cluster(
        topology=Topology.EENET,
        memory_range=(node_mem, node_mem+1), 
        channel_range=(1000, 1001)
        )
    demand = RandomDemand(qubit_num, 0.5, gate_range)

    time_limit = 30
    start = time.time()
    cmf = CMF(cluster, demand, solver_time=time_limit, silent=True)
    # cmf = CMF(cluster, demand,silent=False)
    cmf.build()
    obj = cmf.solve()
    end = time.time()
    print(obj, end-start)

    start = time.time()
    cmp = CMP(cluster, demand, k=5, solver_time=time_limit, silent=True)
    # cmp = CMP(cluster, demand, k=10, silent=False)
    cmp.build()
    obj = cmp.solve()
    end = time.time()
    print(obj, end-start)

    assign_f, util_f = cmf.get_results()
    assign_p, util_p = cmp.get_results()

    print(assign_f)
    print(assign_p)

    print(util_f)
    print(util_p)

    prices = { edge: 1/cluster.graph.edges[edge]['channel_prob'] for edge in cluster.graph.edges }
    cost_f = sum([util_f[edge]*prices[edge] for edge in util_f])
    cost_p = sum([util_p[edge]*prices[edge] for edge in util_p])

    print(cost_f)
    print(cost_p)