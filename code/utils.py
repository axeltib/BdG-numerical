import numpy as np

def get_hamiltonian(n, t=1, mu=1):
    """ Returns the hopping hamiltonian (tight binding) with hopping matrix element t and chemical potential mu. THe number of sites is denoted by n. """
    t_array = t/2 * np.ones(n-1)
    return - (np.diag(t_array, 1) + np.diag(t_array, -1))  - mu * np.eye(n)

def get_kiatev_hamiltonian(n, t=1, delta=1, mu=0):
    """ Returns the BdG Hamiltonian for a spinless chain model. """
    # pauli matrices
    H = np.zeros((2*n, 2*n), dtype=complex)

    pauli_z = np.array([[1,0],[0,-1]])
    pauli_y = np.array([[0,-1j],[1j,0]])

    t_array = off_diag_block(np.copy(H), t * pauli_z, n, 2) + off_diag_block(np.copy(H), t * pauli_z, n, -2)
    delta_array =  off_diag_block(np.copy(H), delta * 1j * pauli_y, n, 2) + off_diag_block(np.copy(H), -delta * 1j * pauli_y, n, -2)
    mu_array = off_diag_block(np.copy(H), mu * pauli_z, n, 0)

    return -(t_array + delta_array + mu_array)

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
    if T == 0:
        return 0
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

def off_diag_block(H, a, n, offset):
    """ Used for creating off-diagonal block matrices with 2x2 matrices as blocks. """
    if offset > 0:
        range_settings = (0, n+1, 2)
    elif offset == 0:
        range_settings = (0, n+3, 2)
    else:
        range_settings = (0-offset, n+1-offset, 2)

    for i in range(range_settings[0], range_settings[1], range_settings[2]):
        H[i+offset,i] = a[0,0]
        H[i+offset,i+1] = a[0,1]
        H[i+offset+1,i] = a[1,0]
        H[i+offset+1,i+1] = a[1,1]

    return H
