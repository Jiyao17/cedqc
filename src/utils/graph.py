
from typing import NewType

import networkx as nx

# nodes are named by strings in networkx library
Node = NewType('NodeName', str)
Pair = NewType('Pair', tuple[Node, Node])
Path = NewType('Path', tuple[Node])
Paths = NewType('Paths', list[Path])
PairKSP = NewType('PairKSP', dict[Pair, Paths])


def YenKSP(
        graph: nx.Graph, 
        pairs: list[Pair], 
        k: int=1, 
        weight: str=None, 
        existing_paths: dict=None
        ):
    """
    find k shortest paths between all given pairs
    Parameters
    ----------
    graph: nx.Graph
        The network graph
    pairs: list[Pair], optional
        The pairs of nodes to find paths between
    k: int, optional (default=1)
        The number of shortest paths to find between each pair
    weight: str, optional (default=None)
        -None: least hops
        -'length': Shortest path length
    existing_paths: dict, optional. 
        If provided, those paths are included in the k shortest paths.
    """
    APYenKSP: PairKSP = { pair: [] for pair in pairs }
    
    if existing_paths is not None:
        for pair in existing_paths.keys():
            # slice the existing paths up to k
            APYenKSP[pair] += existing_paths[pair][:k]
    
    for pair in pairs:
        path_iter = nx.shortest_simple_paths(graph, *pair, weight=weight)
        while len(APYenKSP[pair]) < k:
            try:
                path: Path = tuple(next(path_iter))
                APYenKSP[pair].append(path)
            except StopIteration:
                break

    return APYenKSP

