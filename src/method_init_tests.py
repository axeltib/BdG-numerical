import pickle
import argparse

import numpy as np

from bdg_method import BdG_method
from cluster_bdg_method import CBdG_method

# set energy unit
t = 1

    
def main():
    N = 20
    Nc = 5
    
    t = 1
    
    T = 0.1 * t
    V = 3.0
    
    # generate mu
    mean, var = 0, 0.1 # mean and standard deviation

    mu_array = 1.0 * np.random.normal(mean, var, N) * t

    normal_method = BdG_method(N, 0, mu_array, t, V, T, 1e-9)
    cluster_method = CBdG_method(N, Nc, mu_array, t, V, T, 1e-9)
    
    normal_method.run_solver()
    cluster_method.run_solver()
    
    
if __name__ == "__main__":
    main()
