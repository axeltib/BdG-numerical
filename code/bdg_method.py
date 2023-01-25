import numpy as np

from utils import get_bdg_hamiltonian

class BdG_method:

    """ Used to run BdG numerical solution to BCS problems.     """
    def __init__(self, N: int, Nc: int, mu: float, t: float, T: float, orb_wave_function, spatial_dims=1) -> None:
        # Simulation parameters
        self.N = N
        self.Nc = Nc
        
        self.mu = mu
        self.t = t
        self.T = T
        
        self.orbital_wave_function = orb_wave_function
        
        self.convergence_threshold = 1e-4
        
        # Physical constants
        self.kB = 1
        
        # Initialize quantities
        self.H0 = self.get_hamiltonian()
        self.delta = self._initialize_op()
        self.H_bdg = get_bdg_hamiltonian(self.H0, self.delta)
        self.spatial_dimensions = spatial_dims  #Relevant in the calculation of the spatial gap parameter 
        
        
    def get_hamiltonian(self):
        """ Returns the hopping hamiltonian (tight binding) with hopping matrix element t and chemical potential mu. THe number of sites is denoted by n. """
        def get_H_element(i, j):
            if i == j:
                return - self.mu
            elif abs(i - j) == 1:
                return - self.t
            else:
                return 0
        H = np.array([get_H_element(i, j) for i in range(self.N) for j in range(self.N)]    )
        return np.array(H, dtype=np.complex64).reshape((self.N,self.N))

        
    def _initialize_op(self):
        """ Generates the initial gap/order parameter. """
        return (2*np.random.random((self.N,self.N)) - 1) + 1j*(2*np.random.random((self.N,self.N)) - 1)


    def fermi_dirac_distribution(self, E: float):
        """ Calculates the Fermi-Dirac distribution given the energy. """
        if self.T == 0:
            return 0
        return 1 / (np.exp(np.real(E)/(self.kB * self.T)) + 1)


    def self_consistent_condition(self, eigen_vectors, energy_eigenvalues):
        """ Updates the gap parameter according to the self-consistent condition. """
        V = 1
        self.delta[:,:] = 0
        # Divide the eigenevectors into its u and v part, by splitting between middle of the rows
        u_eigenvectors = eigen_vectors[:self.N,:]
        v_eigenvectors = eigen_vectors[self.N:,:]

        for i, E in enumerate(energy_eigenvalues):
            if E < 0:
                continue
            u = u_eigenvectors[:,i]
            v = v_eigenvectors[:,i]
            self.delta += V * np.outer(u, v.conj()) * (1 - 2*self.fermi_dirac_distribution(E))
        
    
    def get_spatial_1d_gap_parameter(self):
        """ Calculates and returns the 1D spatial gap parameter input. """
        delta_j = np.zeros(self.N, dtype='complex64')
        for j in range(self.N):
            for beta1 in range(self.N):
                for beta2 in range(self.N):
                    delta_j[j] += self.delta[beta1,beta2] \
                        * self.orbital_wave_function(j, beta1, self.N) *  np.conj(self.orbital_wave_function(j, beta2, self.N))
        return delta_j
    
    
    def get_order_parameter_amplitude(self, delta_spatial: np.array):
        """ Calculates the amplitude of the order parameter, which is used to measure the gap. """
        return abs(min(np.abs(delta_spatial)) - max(np.abs(delta_spatial)))
    
    
    def run_solver(self) -> int:
        """ Runs the main loop of the solver, does this n times. """
        iterations = 0
        last_amplitude = self.get_order_parameter_amplitude(self.get_spatial_1d_gap_parameter()) + 1
        while abs(self.get_order_parameter_amplitude(self.get_spatial_1d_gap_parameter()) - last_amplitude) > self.convergence_threshold:
            last_amplitude = self.get_order_parameter_amplitude(self.get_spatial_1d_gap_parameter())
            energy_array, eigen_vectors = np.linalg.eig(self.H_bdg)
            self.self_consistent_condition(eigen_vectors, energy_array)  # Updates the gap parameter
            self.H_bdg = get_bdg_hamiltonian(self.H0, self.delta)
            print(abs(self.get_order_parameter_amplitude(self.get_spatial_1d_gap_parameter()) - last_amplitude))
            iterations += 1
            
        return iterations
