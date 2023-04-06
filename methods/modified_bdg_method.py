import numpy as np

from bdg_method import BdG_method
from utils import get_bdg_hamiltonian

class MBdG_method(BdG_method):
    """
    Assumes only 1D. 
    """
    def __init__(self, N: int, Nc: int, mu: float, t: float, T: float, numiter: int, spatial_dims=1, delta=0) -> None:
            super().__init__(N, Nc, mu, t, T, numiter, spatial_dims=spatial_dims, delta=delta)
            if 2*self.Nc >= self.N:
                raise Exception("Clustering number Nc must be less than N!")
        
        
    def construct_cluster_matrices(self, site_index):
        """ Used to find the lattice neighbor elements in the BdG Hamiltoinan (H and delta). 
        Should use Nc. """
        # look Nc around the site index
        cluster_H =         np.zeros((2*self.Nc+1, 2*self.Nc+1), dtype=self.dtype)
        cluster_delta =  np.zeros((2*self.Nc+1, 2*self.Nc+1), dtype=self.dtype)
        
        index_list = [i%self.N for i in range(site_index-self.Nc, site_index+self.Nc+1, 1)]
        
        for cluster_i, site_i in enumerate(index_list):
            for cluster_j, site_j in enumerate(index_list):
                cluster_H[cluster_i, cluster_j] = self.H0[site_i, site_j]
                cluster_delta[cluster_i, cluster_j] = self.delta[site_i, site_j]
        return cluster_H, cluster_delta
        
                
    def update_delta_site(self, site_i):
        """ Updates the gap parameter according to the self-consistent condition. """
        cluster_H, cluster_delta = self.construct_cluster_matrices(site_i)
        cluster_Hbdg = get_bdg_hamiltonian(cluster_H, -cluster_delta)
        energy_eigs, eigen_vecs = np.linalg.eig(cluster_Hbdg)
              
        # Divide the eigenevectors into its u and v part, by splitting between middle of the rows
        u_eigenvectors = eigen_vecs[:2*self.Nc+1,:]
        v_eigenvectors = eigen_vecs[2*self.Nc+1:,:]

        if u_eigenvectors.shape != v_eigenvectors.shape:
            raise Exception("The eigenvectors u and v must have the same shape!")

        delta_diag = np.zeros(2*self.Nc+1, dtype=self.dtype)
        
        # for j in range(self.Nc):
        for i, E in enumerate(energy_eigs):
            if E < 0:  # should disregard negative eigenvalues, they
                continue
            u = u_eigenvectors[:,i]
            v = v_eigenvectors[:,i]
            delta_diag +=  self.V * u * v.conj() * (1 - 2*self.fermi_dirac_distribution(E)) #elem vise mult
            
        return np.mean(delta_diag)
        
        
    def run_one_pass(self):
        """ Runs one pass over the sites to update the delta"""
        delta_tmp = np.zeros(self.N, dtype=self.dtype)
        for i in range(self.N):
            delta_tmp[i] = self.update_delta_site(i)
        self.delta = np.diag(delta_tmp)
        
    def run_solver(self):
        """ Runs the solver. """
        for i in range(self.num_iterations):
            self.run_one_pass()
        return self.delta
    

