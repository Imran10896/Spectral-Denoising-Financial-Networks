import cpnet

def cp_centralization(G, alg):
    """
    Fast cp-centralization computation.
    Avoids pandas conversion (very slow in loops).
    """
    alg.detect(G)

    # Direct numpy conversion (FAST)
    x = np.fromiter(alg.get_coreness().values(), dtype=float)

    n = G.number_of_nodes()
    if n <= 2:
        return np.nan

    C = 1 - (2 / (n - 2)) * (np.sum(x) - 1)
    return C
