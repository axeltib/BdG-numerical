from datetime import datetime

import numpy as np

from src.utils.job_utils import *

from src.utils.result import TimingResult

""" The goal here is to run the job for one temperature with multiple system
sizes. 


"""

# set energy unit
t = 1
    
def main():
    results = TimingResult()
    
    # Define study variables
    args = parse_arguments()
    
    args["V"] = 3.0 * t
    args["temperature"] = float(args["temperature"])
    args["convergence_threshold"] = 1e-9
    
    for N in args["NumSites"]:
        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        print("{0} Running sim for N={1}".format(dt_string, N))
        
        mu_array = get_mu_array(
            0, 
            0, 
            args["bias"], 
            N
        )
        
        # print out to log
        print("[main] Job arguments: ")
        for key, val in args.items():
            print(key, val)


        run_result = run_methods(
            t,
            N, 
            args["V"], 
            mu_array, 
            args["temperature"], 
            args["ClusterSizes"], 
            args["convergence_threshold"]
        )
        
        results.add_run_result(N, run_result)
    
    pickle_results(
        args["JobId"] + "-timing.pickle", 
        results
    )
    
    print("[main] Success!")
    
    
if __name__ == "__main__":
    main()
