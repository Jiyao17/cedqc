

import numpy as np
import gurobipy as gp

from .solver import Solver
from ..network import Cluster, Topology
from ..task import Demand, RandomDemand


class CMF(Solver):
    def __init__(self, 
        cluster: Cluster, demand: Demand,
        solver_time: float=60, 
        silent: bool=True
        ) -> None:
        super().__init__(cluster, demand)

        # all 3-tuple combinations of nodes
        self.triples = []
        for i, j in self.pairs:
            for v in self.nodes:
                if v != i and v != j:
                    self.triples.append((i, v, j))
        
        self.flow_triples = []
        for i, v, j in self.triples:
            in_pair = (i, j)
            if (i, v) in self.pairs:
                out_pair1 = (i, v)
            else:
                out_pair1 = (v, i)
            if (v, j) in self.pairs:
                out_pair2 = (v, j)
            else:
                out_pair2 = (j, v)

            self.flow_triples.append((out_pair1, out_pair2, in_pair))

        
        self.model = gp.Model("CMF")
        self.model.setParam('TimeLimit', solver_time)
        if silent:
            self.model.setParam('OutputFlag', 0)

    def build(self,):
        self.add_vars()
        self.add_constrs()
        self.set_obj()

    def add_vars(self,):
        # flows
        self.f = {}
        for out_pair1, out_pair2, in_pair in self.flow_triples:
            self.f[(out_pair1, in_pair)] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f'f_{out_pair1}_{in_pair}')
            self.f[(out_pair2, in_pair)] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f'f_{out_pair2}_{in_pair}')
            self.model.addConstr(self.f[(out_pair1, in_pair)] >= 0)
            self.model.addConstr(self.f[(out_pair2, in_pair)] >= 0)


        # edge utilization
        self.phi = {}
        for edge in self.edges:
            self.phi[edge] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f'phi_{edge}')
            self.model.addConstr(self.phi[edge] >= 0)

        # qubit assignment
        self.y = {}
        for qubit in self.demand.qubits:
            for node in self.nodes:
                self.y[(qubit, node)] = self.model.addVar(
                    vtype=gp.GRB.BINARY, name=f'y_{qubit}_{node}')
                
    def add_constrs(self,):
        # equal contribution to swappings
        for out_pair1, out_pair2, in_pair in self.flow_triples:
            self.model.addConstr(self.f[(out_pair1, in_pair)] == self.f[(out_pair2, in_pair)])

        # in-flow
        self.inflow = {}
        for pair in self.pairs:
            # in-flow
            if pair not in self.inflow:
                self.inflow[pair] = 0
            if pair in self.edges:
                self.inflow[pair] += self.phi[pair]

            for out_pair1, out_pair2, in_pair in self.flow_triples:
                if in_pair == pair:
                    o1, o2 = out_pair1
                    v = o1 if o1 not in in_pair else o2
                    self.inflow[pair] += self.nodes[v]['swap_prob'] * 0.5 \
                        * (self.f[(out_pair1, pair)] + self.f[(out_pair2, pair)])

        # out-flow
        self.outflow = {}
        for pair in self.pairs:
            self.outflow[pair] = 0
            for out_pair1, out_pair2, in_pair in self.flow_triples:
                if out_pair1 == pair:
                    out_pair = out_pair1
                elif out_pair2 == pair:
                    out_pair = out_pair2
                else:
                    continue

                # out-flow
                self.outflow[pair] += self.f[(out_pair, in_pair)]


        # flow conservation
        self.flow_conserve = {}
        demands = {(i, j): 0 for i, j in self.pairs}
        for i, j in self.pairs:
            self.flow_conserve[(i, j)] = self.inflow[(i, j)] - self.outflow[(i, j)]
            
            demands[(i, j)] += gp.quicksum(c_ab * self.y[(a, i)] * self.y[(b, j)] 
                for (a, b), c_ab  in self.demand.demands.items())
            demands[(i, j)] += gp.quicksum(c_ab * self.y[(a, j)] * self.y[(b, i)] 
                for (a, b), c_ab  in self.demand.demands.items())
            self.model.addConstr(self.flow_conserve[(i, j)] >= demands[(i, j)])
            self.model.addConstr(self.flow_conserve[(i, j)] >= 0)
            
        # qubit assignment
        for qubit in self.demand.qubits:
            self.model.addConstr(
                gp.quicksum(self.y[(qubit, node)] for node in self.nodes) == 1)
        
        # memory constraints
        for node in self.nodes:
            self.model.addConstr(
                gp.quicksum(self.y[(qubit, node)] for qubit in self.demand.qubits) \
                <= self.nodes[node]['memory'])
            
        # edge capacity constraints
        for edge in self.edges:
            cap = self.cluster.graph.edges[edge]['capacity']
            self.model.addConstr(self.phi[edge] <= cap)

    def set_obj(self,):
        # set objective as weighted sum of edge utilization
        edges = self.cluster.graph.edges(data=True)
        cost = gp.quicksum(1/data['channel_prob'] * self.phi[(i, j)] 
            for i, j, data in edges)
        # cost = gp.quicksum(self.phi[edge] for edge in self.edges)

        self.model.setObjective(cost, gp.GRB.MINIMIZE)
        self.model.update()

    def get_results(self,):
        # find qubit assignment and edge utilization
        assignment: 'dict[str, list[int]]' = {}
        for node in self.nodes:
            assignment[node] = []
            for qubit in self.demand.qubits:
                if self.y[(qubit, node)].x != 0:
                    assignment[node].append(qubit)
        # filter out nodes without qubit assignment
        assignment = {node: qubits for node, qubits in assignment.items() if len(qubits) > 0}

        utilization = {}
        for edge in self.edges:
            if self.phi[edge].x != 0:
                utilization[edge] = self.phi[edge].x

        return assignment, utilization


if __name__ == "__main__":
    node_mem = 16
    qubit_num = 64
    gate_range = (8, 16)
    cluster = Cluster(
        topology=Topology.EENET,
        memory_range=(node_mem, node_mem+1), 
        channel_range=(1000, 1001)
        )
    cluster.plot()
    demand = RandomDemand(qubit_num, 0.2, gate_range)

    cmf = CMF(cluster, demand)
    cmf.build()
    cmf.solve()

    # print(sum([y.x for y in cmf.y.values() if y.x != 0]))
    # print(demand.demands)


    for node in cmf.nodes:
        assignment = []
        for qubit in demand.qubits:
            if cmf.y[(qubit, node)].x != 0:
                assignment.append(qubit)
        if len(assignment) > 0:
            print(node, assignment)

    for edge in cmf.edges:
        if cmf.phi[edge].x != 0:
            print(f'phi_{edge}: {cmf.phi[edge].x}')


