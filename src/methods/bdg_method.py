import time

import numpy as np

from src.methods.utils import get_bdg_hamiltonian
from src.utils.result import SimResult
from src.utils.timing import get_current_time_in_ms

class BdG_method:
    """ Used to run BdG numerical solution to BCS problems.     """
    def __init__(self, N: int, Nc: int, mu: np.array, t: float, V: float, T: float, conv_thrs: float) -> None:
        
        self.kB = 1
        
        # Simulation parameters
        self.N = N
        self.Nc = Nc
        
        self.mu = mu
        self.t = t
        self.temperature = T * self.t
        self.V = V
        
        self.convergence_threshold = conv_thrs
        
        self.dtype = "complex64"
        # Physical constants
        
        # Initialize quantities
        self.H0 = self.get_hamiltonian()

        self.delta = self._initialize_op()
        self.last_global_delta = self.get_global_delta() + 2*self.convergence_threshold

        self.H_bdg = get_bdg_hamiltonian(self.H0, self.delta)
        
        # Storage for the results
        self.id = self._set_id()
        self.results = SimResult()
        
        
    def _set_id(self) -> None:
        self.id = "NBM"

    def get_id(self) -> str:
        return self.id

    def get_hamiltonian(self):
        """ Returns the hopping hamiltonian (tight binding) with hopping matrix element t and chemical potential mu. THe number of sites is denoted by n. """
        def get_H_element(i, j, n):
            if i == j:
                if i == n//2:
                    return - self.mu[i]
                else:
                    return - self.mu[i]
            elif abs(i - j) == 1:
                return -self.t
            elif (i == n-1 and j == 0) or (j == n-1 and i == 0):
                return -self.t
            else:
                return 0
        H = np.array([get_H_element(i, j, self.N) for i in range(self.N) for j in range(self.N)]    )
        return np.array(H, dtype=self.dtype).reshape((self.N,self.N))

        
    def _initialize_op(self) -> np.array:
        """ Generates the initial gap/order parameter. """
        # Deterministic way of initializing
        # diag_elements = [np.sqrt(2) + np.sqrt(2)*1j] * self.N 
        
        # Randomized way of initializing; on complex unit circle. 
        diag_elements = np.exp(1j * np.random.uniform(0, np.pi, self.N))
        
        #diag_elements = np.random.random(self.N)  + 1j*np.random.random(self.N) 
        #return (2*np.random.random((self.N,self.N)) - 1) + 1j*(2*np.random.random((self.N,self.N)) - 1)
        return np.diag(diag_elements)

    def fermi_dirac_distribution(self, E: float):
        """ Calculates the Fermi-Dirac distribution given the energy. """
        if self.temperature == 0:
            return 0
        real_E = np.real(E)
        return 1 / (np.exp(real_E / (self.kB * self.temperature)) + 1)


    def self_consistent_condition(self):
        """ Updates the gap parameter according to the self-consistent condition. """
        energy_eigs, eigen_vecs = np.linalg.eig(self.H_bdg)
        # Divide the eigenevectors into its u and v part, by splitting between middle of the rows
        u_eigenvectors = eigen_vecs[:self.N,:]
        v_eigenvectors = eigen_vecs[self.N:,:]
        delta_diag = np.zeros(self.N, dtype=self.dtype)
        
        for i, E in enumerate(energy_eigs):
            if E < 0:  # should disregard negative eigenvalues, they
                continue
            u = u_eigenvectors[:,i]
            v = v_eigenvectors[:,i]
            delta_diag +=  -self.V * u * v.conj() * (1 - 2*self.fermi_dirac_distribution(E)) #elem vise mult
            #delta_tmp += self.V * np.outer(u, v.conj()) * (1 - 2*self.fermi_dirac_distribution(E))
        return np.diag(delta_diag) / self.t

    def set_temperature(self, T: float):
        """ Sets the temperature of the solver. Used when whanting to solve the same system for multiple temperatures. """
        self.temperature = T*self.t
        
        
    def get_global_delta(self) -> float:
        return np.mean(self.get_delta())
        
    def get_delta(self):
        return np.abs(np.diag(self.delta))
    
    def check_if_sim_complete(self) -> bool:
        iteration_diff = abs(self.get_global_delta() - self.last_global_delta)
        return iteration_diff < self.convergence_threshold
        
    def run_one_pass(self):
        self.last_global_delta = self.get_global_delta()
        
        self.delta = self.self_consistent_condition()  # Updates the gap parameter
        self.H_bdg = get_bdg_hamiltonian(self.H0, self.delta)
                
        
    def run_solver(self):
        """ Runs the main loop of the solver, does this n times. """
        start_time = get_current_time_in_ms()
        
        while self.check_if_sim_complete() != True:
            time_prev = get_current_time_in_ms()
            self.run_one_pass()
            # Update results
            self.results.append_time_per_iteration(
                abs(get_current_time_in_ms() - time_prev)
            )
            self.results.increment_iteration_count()
        
        self.results.set_total_time(
            abs(get_current_time_in_ms() - start_time)
        )
        
        self.results.set_energy_gap(self.get_delta())    

        return self.results
