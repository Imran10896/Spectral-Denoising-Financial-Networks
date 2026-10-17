import numpy as np
import pandas as pd

def rmt(df):

    # --- Standardize ---
    returns = (df - df.mean(axis=0)) / df.std(axis=0, ddof=1)
    
    if hasattr(returns, "values"):
        
        returns = returns.values

    T, N = returns.shape

    # --- RMT condition ---
    if T <= N:
        raise ValueError("RMT requires T > N")

    # --- Correlation ---
    C_FULL = np.corrcoef(returns, rowvar=False)
    C_FULL = np.nan_to_num(C_FULL)

    # --- Eigen decomposition ---
    eigvals, eigvecs = np.linalg.eigh(C_FULL)

    # --- MP bounds ---
    Q = T / N
    lam_minus = (1 - np.sqrt(1 / Q))**2
    lam_plus  = (1 + np.sqrt(1 / Q))**2

    # --- Structured ---
    idx_str = eigvals > lam_plus
    C_STR = eigvecs[:, idx_str] @ np.diag(eigvals[idx_str]) @ eigvecs[:, idx_str].T

    # --- Random ---
    idx_rnd = eigvals <= lam_plus
    C_RND = eigvecs[:, idx_rnd] @ np.diag(eigvals[idx_rnd]) @ eigvecs[:, idx_rnd].T
    
    C_STR = (C_STR + C_STR.T)/2
    C_RND = (C_RND + C_RND.T)/2
    
        # --- Zero diagonal ---
    np.fill_diagonal(C_FULL, 0)
    np.fill_diagonal(C_STR, 0)
    np.fill_diagonal(C_RND, 0)


    return C_FULL, C_STR, C_RND
