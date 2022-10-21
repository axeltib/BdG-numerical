import numpy as np

def get_hamiltonian(n, t=1, mu=1):
    return -t - mu * np.eye(n)

# Should the order parameter be its own class?
def generate_initial_order_parameter(n):
    return (2*np.random.random((n,n)) - 1) + 1j*(2*np.random.random((n,n)) - 1)

def get_bdg_hamiltonian(H, delta):
    n = np.shape(H)[0]
    block_row1 = np.concatenate((H, delta), axis=0)
    block_row2 = np.concatenate((delta.T.conj(), -H.conj()), axis=0)

    bdg_hamiltonian = np.concatenate((block_row1, block_row2), axis=1)
    assert np.shape(bdg_hamiltonian) == (2*n, 2*n)
    
    return bdg_hamiltonian

def fm_distribution(E, T, kB = 1):
    # Take only the real part of the energy (as it can be complex)
    return 1 / (np.exp(np.real(E)/(kB*T)) + 1)

def update_delta(N, delta, energy_values, eigen_vectors, V=1.2, T=1):
    delta[:,:] = 0

    # Divide the eigenevectors into its u and v part, by splitting between middle of the rows
    u_eigenvectors = eigen_vectors[:N,:]
    v_eigenvectors = eigen_vectors[N:,:]

    # Sum over the eigenstates
    for i, E in enumerate(energy_values):
        u = u_eigenvectors[:,i]
        v = v_eigenvectors[:,i]
        
        delta += np.outer(v.conj(), u)*(1 - 2*fm_distribution(E, T))  - np.outer(u, v.conj()) * (1 - 2*fm_distribution(E, T))
    
    return V/2 * delta

