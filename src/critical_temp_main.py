import pickle
import argparse

import numpy as np

from utils import find_critical_temperature

from bdg_method import BdG_method
from cluster_bdg_method import CBdG_method

# set energy unit
t = 1

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--ClusterSizes", help="The cluster sizes of methods, each arguments yields one method.", nargs="*",type=int)
    parser.add_argument("-N", "--NumSites", help="The size of the system in question, or the number of sites used. ", type=int)
    parser.add_argument("-i", "--JobId", help="The job id in order to keep track of output files and logs.")

    args = parser.parse_args()
    return vars(args)

def pickle_results(id, Tc_lists, delta0_lists):
    with open(id+"-crit_temps.pickle", 'wb') as f:
        pickle.dump(Tc_lists, f)
        
    with open(id+"-delta0s.pickle", 'wb') as f:
        pickle.dump(delta0_lists, f)


def run_methods(N, V, mu_set, T_set, cluster_sizes, convergence_threshold):
    n_methods = len(cluster_sizes) + 1
    
    Tc_lists = [[] for i in range(n_methods)]
    delta0_lists = [[] for i in range(n_methods)]
    
    for mu in mu_set:
        solvers = [BdG_method(N, 0, mu, t, V, T_set[0], convergence_threshold)]
        for Nc in cluster_sizes:
            solvers.append(CBdG_method(N, Nc, mu, t, V, T_set[0], convergence_threshold))

        print("Mu = {0}".format(mu), end=": ")

        deltas = [[] for i in range(n_methods)]
        
        for T in T_set:
            print(round(T, 2), end=" ")
            
            for i, solver in enumerate(solvers):
                solver.set_temperature(T)
                solver.run_solver()
                deltas[i].append(solver.get_global_delta())
            
        for i in range(n_methods):
            Tc_lists[i].append(find_critical_temperature(T_set, deltas[i]))
            delta0_lists[i].append(deltas[i][0])

        print("\n")
                
    return Tc_lists, delta0_lists

    
def main():
    # Define study variables
    args = parse_arguments()
    args["V"] = 3.0 * t
    
    args["T_set"] = np.linspace(0, 0.6, 40)
    args["mu_set"] = np.linspace(0.0, 3.0, 20) * t
    
    args["convergence_threshold"] = 1e-8

    # print out to log
    print("Job arguments: ")
    for key, val in args.items():
        if key == "T_set":
            print("T_set. Min: {0}, max: {1}, num: {2}".format(min(args["T_set"]), max(args["T_set"]), len(args["T_set"])))
        elif key == "mu_set":
            print("mu_set. Min: {0}, max: {1}, num: {2}".format(min(args["mu_set"]), max(args["mu_set"]), len(args["mu_set"])))
        else: 
            print(key, val)

    Tc_lists, delta0_lists = run_methods(
        args["NumSites"], 
        args["V"], 
        args["mu_set"], 
        args["T_set"], 
        args["ClusterSizes"], 
        args["convergence_threshold"]
    )
    
    pickle_results(args["JobId"], Tc_lists, delta0_lists)
    
    print("[Python] Success!")
    
    
if __name__ == "__main__":
    main()
