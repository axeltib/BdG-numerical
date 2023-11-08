import numpy as np

from src.utils.job_utils import *

# set energy unit
t = 1
    
def main():
    # Define study variables
    args = parse_arguments()

    args["V"] = 3.0 * t
    
    args["mu_set"] = get_mu_array(
        args["amplitude"], 
        args["frequency"], 
        args["bias"], 
        args["NumSites"]
    )
    
    args["convergence_threshold"] = 1e-12
    
    # print out to log
    print("[main] Job arguments: ")
    for key, val in args.items():
        print(key, val, type(val))

    args["temperature"] = float(args["temperature"])
    
    run_result = run_methods(
        t, 
        args["NumSites"], 
        args["V"], 
        args["mu_set"], 
        args["temperature"], 
        args["ClusterSizes"], 
        args["convergence_threshold"]
    )
    
    pickle_results(
        args["JobId"] + "-inhom-runresult.pickle", 
        run_result
    )
    
    print("[main] Success!")
    
    
if __name__ == "__main__":
    main()
