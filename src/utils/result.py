import numpy as np

class System:
    """ System description. """
    def __init__(self, N: int, T: float, is_hom, bias=1.0, freq=0.0, amp=0.0) -> None:
        self.N = N
        self.temp = T
        self.bias = bias
        self.is_hom = is_hom
        
        self.freq = freq
        self.amp = amp

class SimResult:
    """ Encapsulates the different types of simulation results from one 
    simulation.
    
    Currently supported:
    - on site energy gap
    - time per iteration
    
    """

    def __init__(self) -> None:
        # Expected to contain a list of lists with spatial energy gaps.
        self.final_delta = np.array
        # Time stored in milliseconds
        self.time_per_iteration = np.array([]) #ms
        self.iteration_count = 0 
        
        self.total_time = 0 #ms
        
    def get_delta(self) -> np.array:
        return self.final_delta
    
    def get_time_per_iteration(self) -> np.array:
        return self.time_per_iteration
    
    def get_num_of_iterations(self) -> int:
        return self.iteration_count
    
    def set_energy_gap(self, delta: np.array) -> None:
        self.final_delta = delta
        return 
        
    def set_total_time(self, time: float) -> None:
        """ Sets the total time in ms. """
        self.total_time = time #ms
        
    def append_time_per_iteration(self, t: float) -> None:
        if t < 0:
            raise Exception("Time appended must be positive and non-zero.")
        self.time_per_iteration = np.append(self.time_per_iteration, t)
        return
        
    def increment_iteration_count(self) -> None:
        self.iteration_count += 1
        return 
        
    
class RunResults:
    """ Stores the results of a run with multiple simulations that 
    one want to compare results against. """
    
    def __init__(self) -> None:
        self.sim_results = dict()
        
    def add_result(self, id: str, result: SimResult) -> None:
        self.sim_results[id] = result
        
        
class TimingResult:
    """ Store the result of runs with multiple system sizes.
    """
    
    def __init__(self) -> None:
        self.run_results = dict()
    
    def add_run_result(self, num_sites: int, result: RunResults):
        self.run_results[num_sites] = result