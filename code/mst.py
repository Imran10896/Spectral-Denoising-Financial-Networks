import numpy as np
import pandas as pd
import networkx as nx

def mst(corr):
    """
    Build MST from correlation matrix using Mantegna distance.
    """
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    D = np.sqrt(2 * (1 - corr))

    G = nx.from_numpy_array(D)
    MST = nx.minimum_spanning_tree(G, weight="weight")

    return MST
