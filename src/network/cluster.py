

import os
import enum

import numpy as np
import networkx as nx
import geopy.distance as geo

from .quantum import HWParam, get_edge_capacity, get_photon_survival_rate
from ..utils.plot import plot_nx_graph



class Topology(enum.Enum):
    ATT = 'topology/ATT.graphml'
    GETNET = 'topology/Getnet.graphml'
    # IOWA = 'topology/IowaStatewideFiberMap.graphml'
    IRIS = 'topology/Iris.graphml'

    # CNET = 'topology/NSFCNET.graphml'
    # INS = 'topology/INS.graphml'
    ION = 'topology/ION.graphml'
    MISSOURI = 'topology/Missouri.graphml'
    RENATOR = 'topology/Renater.graphml'
    NOEL = 'topology/Noel.graphml'
    # LOUISANA = 'topology/Louisiana.graphml'
    # CYNET = 'topology/Cynet.graphml'
    EENET = 'topology/EEnet.graphml'

    TRIANGLE = 'topology/Triangle.graphml'
    PAIR = 'topology/Pair.graphml'


class IDGenerator:
    
        def __init__(self, start: int=0) -> None:
            self.start = start - 1
    
        def __iter__(self):
            return self
    
        def __next__(self):
            self.start += 1
            return self.start


class Cluster:

    def __init__(self, 
            topology: Topology =Topology.NOEL,
            hw_params: dict = HWParam,
            memory_range: tuple=(256, 257),
            channel_range: tuple=(256, 257)
            ) -> None:
        super().__init__()
        self.topology = topology
        self.hw_params = hw_params

        self.graph: 'nx.Graph' = nx.Graph()
        self.nid = IDGenerator()

        filename = topology.value
        path = os.path.join(os.path.dirname(__file__), filename)
        self.graph: nx.Graph = nx.read_graphml(path)
        # force simple graph
        self.graph = nx.Graph(self.graph)

        to_remove = []
        for node, data in self.graph.nodes(data=True):
            if 'Latitude' not in data or 'Longitude' not in data:
                to_remove.append(node)
                continue
            lat, lon= data['Latitude'], data['Longitude']

            self.graph.nodes[node]['pos'] = (lat, lon)
            self.graph.nodes[node]['swap_prob'] = HWParam['swap_prob']
        self.graph.remove_nodes_from(to_remove)

        self.update_edges()
        self.set_node_memory(memory_range)
        self.set_edge_channel(channel_range)
    
    def update_edges(self):
        """
        update edge attributes:
            -length
            -channel_capacity
        based on the geodesic distance between nodes
        """

        for u, v, data in self.graph.edges(data=True):
            u_pos = self.graph.nodes[u]['pos']
            v_pos = self.graph.nodes[v]['pos']
            length = geo.distance(u_pos, v_pos).km
            fiber_loss = self.hw_params['fiber_loss']
            photon_rate = self.hw_params['photon_rate']
            channel_capacity = get_edge_capacity(length, photon_rate, fiber_loss)
            channel_prob = get_photon_survival_rate(length, fiber_loss)

            data['length'] = length
            data['channel_capacity'] = channel_capacity
            data['channel_prob'] = channel_prob

    def set_node_memory(self, cap_range: tuple):
        """
        set the node memory
        """
        for node in self.graph.nodes:
            memory = np.random.randint(*cap_range)
            self.graph.nodes[node]['memory'] = memory

    def set_edge_channel(self, channel_range: tuple):
        """
        set the edge channel
        """

        for edge in self.graph.edges:
            ch = np.random.randint(*channel_range)
            cap = self.graph.edges[edge]['channel_capacity']

            self.graph.edges[edge]['channel'] = ch
            self.graph.edges[edge]['capacity'] = ch * cap

    def plot(self, 
            node_label: str='id', 
            edge_label: str='length', 
            filename: str='./result/fig.png'
            ):
        plot_nx_graph(self.graph, node_label, edge_label, filename)


if __name__ == '__main__':
    net = Cluster()
    net.update_edges()
    net.plot()


    
