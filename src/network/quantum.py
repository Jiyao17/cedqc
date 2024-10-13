
import numpy as np


HWParam = {
    'swap_prob': 0.5, # swap probability
    'fiber_loss': 0.2, # fiber loss
    'photon_rate': 1e4, # photon rate
}


def get_photon_survival_rate(length: float, fiber_loss: float):
    prob = 10 ** (-0.1 * fiber_loss * length)
    return prob

def get_edge_capacity(length: float, photon_rate: float, fiber_loss: float):
    # prob for half fiber (detectors are in the middle of edges)
    prob = 10 ** (-0.1 * fiber_loss * (length/2))
    channel_capacity = photon_rate * prob**2
    
    return channel_capacity

def get_edge_length(capacity: float, photon_rate: float, fiber_loss: float):
    """
    get suggested edge length
    """
    # length = -10 * (1/fiber_loss) * 2 * np.log10(np.sqrt(capacity / photon_rate))
    # simplify the formula
    length = -10 * np.log10(capacity / photon_rate) / fiber_loss

    return length



if __name__ == '__main__':
    costs = [1,] * 13
    swap_prob = 0.5

    costs = complete_swap(costs, swap_prob)
    print(costs)
    costs = sequential_swap(costs, swap_prob)
    print(costs)
    costs = relaxed_complete_swap(costs, swap_prob)
    print(costs)

    # cap = get_edge_capacity(100, 1e4, 0.2)
    # print(cap)
    # length = get_edge_length(100, 1e4, 0.2)
    # print(length)