import numpy as np

from utils import get_bdg_hamiltonian

class BdG_method:

    """ Used to run BdG numerical solution to BCS problems.     """
    def __init__(self, N: int, Nc: int, mu: float, t: float, T: float, numiter: int, spatial_dims=1, delta=0) -> None:
        # Simulation parameters
        self.N = N
        self.Nc = Nc
        
        self.mu = mu
        self.t = t
        self.T = T*t
        self.V = 2 * self.t
        
        self.convergence_threshold = 1e-4
        self.num_iterations = numiter
        self.dtype = "complex64"
        # Physical constants
        self.kB = 1
        
        # Initialize quantities
        self.H0 = self.get_hamiltonian()
        if not np.any(delta):
            self.delta = self._initialize_op()
        else:
            self.delta = delta
        self.H_bdg = get_bdg_hamiltonian(self.H0, -self.delta)
        self.spatial_dimensions = spatial_dims  #Relevant in the calculation of the spatial gap parameter 
        
        
    def get_hamiltonian(self):
        """ Returns the hopping hamiltonian (tight binding) with hopping matrix element t and chemical potential mu. THe number of sites is denoted by n. """
        def get_H_element(i, j, n):
            if i == j:
                if i == n//2:
                    return - self.mu
                else:
                    return -self.mu
            elif abs(i - j) == 1:
                return -self.t
            elif (i == n-1 and j == 0) or (j == n-1 and i == 0):
                return -self.t
            else:
                return 0
        H = np.array([get_H_element(i, j, self.N) for i in range(self.N) for j in range(self.N)]    )
        return np.array(H, dtype=self.dtype).reshape((self.N,self.N))

        
    def _initialize_op(self):
        """ Generates the initial gap/order parameter. """
        diag_elements = 1 + np.random.random(self.N)  + 1j*np.random.random(self.N) 
        #return (2*np.random.random((self.N,self.N)) - 1) + 1j*(2*np.random.random((self.N,self.N)) - 1)
        return np.diag(diag_elements)

    def fermi_dirac_distribution(self, E: float):
        """ Calculates the Fermi-Dirac distribution given the energy. """
        if self.T == 0:
            return 0
        return 1 / (np.exp(np.real(E)/(self.kB * self.T)) + 1)


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
            delta_diag +=  self.V * u * v.conj() * (1 - 2*self.fermi_dirac_distribution(E)) #elem vise mult
            #delta_tmp += self.V * np.outer(u, v.conj()) * (1 - 2*self.fermi_dirac_distribution(E))
        self.delta = np.diag(delta_diag)


    def set_temperature(self, T: float):
        """ Sets the temperature of the solver. Used when whanting to solve the same system for multiple temperatures. """
        self.T = T*self.t
        
        
    def get_global_delta(self):
        return np.mean(np.diag(self.delta))/self.t
        
    def run_solver(self):
        """ Runs the main loop of the solver, does this n times. """
        
        for i in range(self.num_iterations):
            self.self_consistent_condition()  # Updates the gap parameter
            self.H_bdg = get_bdg_hamiltonian(self.H0, -self.delta)
        return self.delta