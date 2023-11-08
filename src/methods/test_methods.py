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
    args = {}
    
    args["V"]  = 3.0 * t
    args["temperature"] = 0.1
    args["convergence_threshold"] = 1e-8
    args["NumSites"] = 20
    
    args["bias"] = 0
    
    
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    print(
        "{0} Running sim for N={1}".format(
            dt_string, 
            args["NumSites"]
        )
    )
    
    
    
    mu_array = get_mu_array(
        0, 
        0, 
        args["bias"], 
        args["NumSites"]
    )

    # Define solver
    solver = BdG_method(
        args["NumSites"], 
        5, 
        mu_array, 
        t,
        args["V"] , 
        args["temperature"], 
        args["convergence_threshold"]
    )

    job_results = TimingResult()
    run_results = RunResults()
    # run simulation
    solver.set_temperature(args["temperature"])
    
    simresult = solver.run_solver()
    run_results.add_result(
        solver.get_id(),
        simresult,
    )

    job_results.add_run_result(args["NumSites"], run_results)    
    
    pickle_results(
        "testjob.pickle", 
        run_results
    )
    
    print("[main] Success!")
    
    
if __name__ == "__main__":
    main()


