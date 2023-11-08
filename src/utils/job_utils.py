import argparse
import pickle
from datetime import datetime

import numpy as np

from src.methods.bdg_method import BdG_method
from src.methods.cluster_bdg_method import CBdG_method

from src.utils.result import RunResults

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--ClusterSizes", help="The cluster sizes of methods, each arguments yields one method.", nargs="+",type=int)
    parser.add_argument("-N", "--NumSites", help="The size of the system in question, or the number of sites used. ", nargs="+", type=int)
    parser.add_argument("-i", "--JobId", help="The job id in order to keep track of output files and logs.", nargs=1, type=str)
    parser.add_argument("-T", "--temperature", help="The temperature of the job in question. This should only be specified in case of one temp jobs. ",type=float)
    parser.add_argument("-f", "--frequency", help="Sine wave frequency of the spatial dependence in the chemical potential.", nargs="?",type=float)
    parser.add_argument("-b", "--bias", help="Sine wave bias of the spatial dependence in the chemical potential (in terms of t).", nargs=1,type=float)
    parser.add_argument("-a", "--amplitude", help="Amplitude of the chemical potential sine wave (in terms of t).", nargs="?",type=float)
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    # args_dict["NumSites"] = args_dict["NumSites"][0]
    args_dict["JobId"] = args_dict["JobId"][0]
    
    return args_dict

def pickle_results(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
        
def get_mu_array(amplitude, freq, bias, N):
    if amplitude == None:
        amplitude = 0
    if freq == None:
        freq = 0    
    
    x_arr = np.linspace(0, 1, N)
    return amplitude * np.sin(2 * np.pi * freq * x_arr) + bias
        
def run_methods(t, N, V, mu, T, cluster_sizes, convergence_threshold) -> RunResults:
    solvers = [BdG_method(N, 0, mu, t, V, T, convergence_threshold)]
    for Nc in cluster_sizes:
        solvers.append(CBdG_method(N, Nc, mu, t, V, T, convergence_threshold))
    
    n_methods = len(solvers)
    delta_list = [0 for i in range(n_methods)]

    run_results = RunResults()

    for _, solver in enumerate(solvers):
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        print("{0} Running sim for N={1}, Nc={2}".format(dt_string, N, solver.Nc))
        
        solver.set_temperature(T)
        simresult = solver.run_solver()
        print("solver id: ", solver.get_id())
        run_results.add_result(
            solver.get_id(),
            simresult,
        )
                
    return run_results

def generate_inhom_mu(site_array, middlepoint, wavelength):
    return 
    