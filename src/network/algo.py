

import numpy as np

from .quantum import HWParam
from ..utils.graph import YenKSP, PairKSP



def swap_by_CST(costs: 'list[float]', swap_prob: float):
    """
    conduct swaps by a Complete Swapping Tree
    """
    # tree configure
    leaf_num = len(costs)
    tree_depth = int(np.ceil(np.log2(leaf_num)))

    # leaf numbers in the second deepest and deepest level
    second_deepest_num = 2 ** tree_depth - leaf_num
    deepest_num = leaf_num - second_deepest_num

    # single leaf cost contribution to the total cost
    sd_cost = 1/swap_prob**(tree_depth-1)
    d_cost = 1/swap_prob**tree_depth

    # costs array of all nodes
    costs = [d_cost * costs[i] for i in range(deepest_num)] + [sd_cost * costs[i] for i in range(deepest_num, leaf_num)]
    
    return costs

def swap_by_RCST(costs: 'list[float]', swap_prob: float):
    """
    conduct swaps by a Relaxed Complete Swapping Tree
    does not apply to heterogeneous costs
    """
    # tree configure
    leaf_num = len(costs)
    tree_depth = int(np.ceil(np.log2(leaf_num)))

    # leaf numbers in the second deepest and deepest level
    second_deepest_num = 2 ** tree_depth - leaf_num
    deepest_num = leaf_num - second_deepest_num

    # single leaf cost contribution to the total cost
    sd_cost = 1/swap_prob**(tree_depth-1)
    d_cost = 1/swap_prob**tree_depth

    # costs array of all nodes
    # arrange deepest nodes from left or right randomly
    random_num = np.random.rand()
    if random_num < 0.5:
        costs = [d_cost] * deepest_num + [sd_cost] * second_deepest_num
    else:
        costs = [sd_cost] * second_deepest_num + [d_cost] * deepest_num
    
    
    return costs

def swap_by_SST(costs: 'list[float]', swap_prob: float):
    """
    conduct swaps by a Sequential Swapping Tree
    each edge is a leaf
    each swapping is a branching node
    """

    costs: np.ndarray = np.array(costs, dtype=float)
    for i in range(2, len(costs) + 1):
        costs[:i] /= swap_prob

    costs = costs.tolist()
    return costs

def swap_DP(costs: 'list[float]', swap_prob: float):
    """
    conduct swaps by a Dynamic Programming
    """
    # costs array of all nodes
    cost_mat = np.zeros((len(costs) + 1, len(costs) + 1))
    swap_nodes = np.zeros((len(costs) + 1, len(costs) + 1), dtype=int)
    # initialize mat[i, j] = costs[i]
    # mat[i, j]: the minimum cost of swapping from nodes i to j, inclusive
    for i in range(len(costs)):
        cost_mat[i, i+1] = costs[i]

    # calculate mat[i, j] from shorter paths
    for frac_len in range(2,  len(costs) + 1):
        min_cost = np.inf
        min_node = None
        for i in range(len(costs) - frac_len + 1):
            j = i + frac_len
            for v in range(i + 1, j):
                cost = (cost_mat[i, v] + cost_mat[v, j]) / swap_prob
                if cost < min_cost:
                    min_cost = cost
                    min_node = v
            cost_mat[i, j] = min_cost
            swap_nodes[i, j] = min_node

    return cost_mat, swap_nodes

def prepare_paths(
        graph, pairs, k, weight='length', 
        swap_func: callable=swap_by_CST,
        hw_params: dict=HWParam
        ):

    APYenKSP = YenKSP(graph, pairs, k, weight)
    
    alpha, beta = solve_paths(APYenKSP, swap_func, hw_params)

    return APYenKSP, alpha, beta

def solve_paths(APYenKSP: PairKSP, swap_func, hw_params):
    """
    solve the paths
    """
    swap_prob = hw_params['swap_prob']
    
    # alpha[(u, p, e)] = # of entanglements used on edge e for pair u via path p
    alpha = {}
    for pair in APYenKSP.keys():
        for path in APYenKSP[pair]:
            edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            costs = swap_func([1,] * len(edges), swap_prob)
            for i, edge in enumerate(edges):
                alpha[(pair, path, edge)] = costs[i]

    # beta[(u, p, v)] = # of memory slots used at node v for pair u via path p
    beta = {}
    for pair in APYenKSP.keys():
        for path in APYenKSP[pair]:
            for i, node in enumerate(path):
                if i == 0:
                    beta[(pair, path, node)] = alpha[(pair, path, (node, path[1]))]
                elif i == len(path) - 1:
                    beta[(pair, path, node)] = alpha[(pair, path, (path[-2], node))]
                else:
                    mem_left = alpha[(pair, path, (path[i-1], node))]
                    mem_right = alpha[(pair, path, (node, path[i+1]))]
                    beta[(pair, path, node)] = mem_left + mem_right
                    
    return alpha, beta


if __name__ == '__main__':
    leaves = [1, 2, 1, 2, 3, 3, 2, 1]
    swap_prob = 0.5
    # leaves = [1] * 8
    
    mat, nodes = swap_DP(leaves, swap_prob)
    print(mat)
    print(nodes)

    costs = swap_by_CST(leaves, swap_prob)
    print(costs, sum(costs))

    costs = swap_by_SST(leaves, swap_prob)
    print(costs, sum(costs))
