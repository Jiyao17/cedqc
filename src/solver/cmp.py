
import gurobipy as gp

from .solver import Solver
from ..network import Cluster, Topology
from ..task import Demand, RandomDemand
from ..network.algo import prepare_paths


class CMP(Solver):
    def __init__(self, 
            cluster: Cluster, demand: Demand,
            weight: str=None,
            k: int=100,
            solver_time: float=60,
            silent: bool=True
            ):
        super().__init__(cluster, demand)
        self.weight = weight
        self.k = k

        
        self.APYenKSP, self.alpha, _ = prepare_paths(
            self.cluster.graph,
            self.pairs,
            self.k,
            self.weight
        )
        
        self.model = gp.Model("CMP")
        self.model.setParam('TimeLimit', solver_time)
        if silent:
            self.model.setParam('OutputFlag', 0)

    def build(self):
        self.add_vars()
        self.add_constrs()
        self.set_obj()

    def add_vars(self):
        # x[(u, p)] generate x[(u, p)] entanglements for pair u using path p
        self.x = {}
        for u in self.pairs:
            for p in self.APYenKSP[u]:
                # self.x[(u, p)] = self.model.addVar(vtype=gp.GRB.INTEGER, name=f'x_{u}_{p}')
                # no name to avoid too long name for gurobi
                self.x[(u, p)] = self.model.addVar(vtype=gp.GRB.INTEGER)
                self.model.addConstr(self.x[(u, p)] >= 0)

        self.y = {}
        for qubit in self.demand.qubits:
            for node in self.nodes:
                self.y[(qubit, node)] = self.model.addVar(vtype=gp.GRB.BINARY)

    def add_constrs(self):
        # entanglement demand provision
        for u in self.pairs:
            provision = gp.quicksum(
                self.x[(u, p)] for p in self.APYenKSP[u])
            
            demand = 0
            i, j = u
            for (a, b), c_ab in self.demand.demands.items():
                demand += c_ab * self.y[(a, i)] * self.y[(b, j)]
                demand += c_ab * self.y[(a, j)] * self.y[(b, i)]

            self.model.addConstr(provision >= demand)

        # qubit assignment
        for qubit in self.demand.qubits:
            self.model.addConstr(
                gp.quicksum(self.y[(qubit, node)] for node in self.nodes) == 1)

        # memory capacity
        for node in self.nodes:
            self.model.addConstr(
                gp.quicksum(self.y[(qubit, node)] for qubit in self.demand.qubits) 
                <= self.nodes[node]['memory'])
            
        # edge capacity
        usages = { edge: 0 for edge in self.edges }
        for u, p, e in self.alpha.keys():
            if e in usages:
                usages[e] += self.alpha[(u, p, e)] * self.x[(u, p)]
            else:
                usages[(e[1], e[0])] += self.alpha[(u, p, e)] * self.x[(u, p)]
        for edge in self.edges:
            self.model.addConstr(usages[edge] 
                <= self.cluster.graph.edges[edge]['channel_capacity'])

    def set_obj(self):
        obj = 0
        for edge in self.edges:
            for u, p, e in self.alpha.keys():
                if e == edge or e == (edge[1], edge[0]):
                    beta = 1/self.cluster.graph.edges[e]['channel_prob']
                    obj += self.alpha[(u, p, e)] * beta * self.x[(u, p)]
        self.model.setObjective(obj, gp.GRB.MINIMIZE)
        self.model.update()

    def get_results(self):
        assignment: 'dict[str, list[int]]' = {}
        for node in self.nodes:
            assignment[node] = []
            for qubit in self.demand.qubits:
                if self.y[(qubit, node)].x > 0.5:
                    assignment[node].append(qubit)
        # filter out nodes without qubit assignment
        assignment = {node: qubits for node, qubits in assignment.items() if len(qubits) > 0}

        utilization = { edge: 0 for edge in self.edges }
        for u, p, e in self.alpha.keys():
            if self.x[(u, p)].x != 0:
                if e in utilization:
                    utilization[e] += self.alpha[(u, p, e)] * self.x[(u, p)].x
                else:
                    utilization[(e[1], e[0])] += self.alpha[(u, p, e)] * self.x[(u, p)].x
        # filter out zero utilization
        utilization = { edge: util for edge, util in utilization.items() if util != 0 }
        
        return assignment, utilization



if __name__ == "__main__":
    node_mem = 2
    qubit_num = 4
    gate_range = (8, 9)
    cluster = Cluster(
        topology=Topology.EENET,
        memory_range=(node_mem, node_mem+1), 
        channel_range=(1000, 1001)
        )
    cluster.plot()
    demand = RandomDemand(qubit_num, 0.2, gate_range)

    cmf = CMP(cluster, demand)
    cmf.build()
    obj = cmf.solve()
    print(obj)

