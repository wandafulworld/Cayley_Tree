import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def bulk_hamiltonian(k,J1=1,J2=1,J3=1):
    """
    Bulk hamiltonian of the 2xLieb-Cayley Tree (UC of the Case 1 Tree)
    Assumes that connectivity K = 4
    :param k: crystal momentum to be iterated over
    :param J1: 1st weak-bond coefficient
    :param J2: 2nd weak-bond coefficient
    :param J3: strong-bond coefficient
    :return: returns the numerical form of the bulk hamiltonian for the given values
    """
    return np.array([[0,J1*1,J3*2*np.exp(-1j*k)],
                     [J1*1,0,J2*1],
                     [J3*2  *np.exp(1j*k),J2*1,0]])


def ssh_positive_band_projector(k):
    return np.array([[0.5,  0.5*np.exp(1j*k)],
                     [0.5*np.exp(-1j*k),    0.5]])

def ssh_negative_band_projector(k):
    return np.array([[0.5,  -0.5*np.exp(1j*k)],
                     [-0.5*np.exp(-1j*k),    0.5]])

def ssh_hamiltonian(k,w=2,v=1): # Top. non trivial phase
    return np.array([[0, v + w*np.exp(1j*k)],
                     [v + w*np.exp(-1j*k),0]])



def spectrum_calculator(hamiltonian,num=1000,**kwargs):
    """
    Numerically calculates the spectrum for a given bulk hamiltonian, by scanning thorugh k-space [0,2pi] and calculating
    eigenvalues for each k-value.
    :param hamiltonian: A bulk hamiltonian that depends on k(!)
    :param num: The number of steps to be taken through the momentum space (higher num = higher discretization)
    :param *args: Allows to pass any other arguments that are fed into the hamiltonian (positionally)
    :return: The discretized spectrum of the bulk hamiltonian
    """
    spectrum = []
    for k in np.linspace(0.01,2*3.14,num):
        evalues = sp.linalg.eigh(hamiltonian(k,**kwargs),eigvals_only=True)
        spectrum.extend(evalues.tolist())
    return spectrum

def band_mapping_calculator(hamiltonian,num=1000,**kwargs):
    """
    Simply returns all eigenvalues and the associated k-value
    :param hamiltonian: function, Bulk hamiltonian
    :param num: int, number of steps between 0 and 2pi
    :param kwargs:
    :return:
    """
    eigenvalues = []
    k_space = []
    for k in np.linspace(0.01,2*3.14,num):
        evalues = sp.linalg.eigh(hamiltonian(k,**kwargs),eigvals_only=True)
        eigenvalues.append(evalues.tolist())
        k_space.append(k)
    return eigenvalues,k_space

def specific_band_projector_calculator(hamiltonian,k,eval_min,eval_max,vector=False,**kwargs):
    """
    Solves the given bulk hamiltonian, then selects the eigenvector whos eigenvalues fits in the min/max eigenvalue window
    supplied and calculates it's outer product.
    :param hamiltonian: a bulk hamiltonian
    :param k: the crystal momentum at which the bulk hamiltonian is supposed to be evaluated
    :param eval_min: the minimum of the eigenvalue window
    :param eval_max: the maximum of the eigenvalue window
    :param vector: If true, function returns the vector instead of the outer product of the vector
    :param args: additional arguments of the bulk hamiltonian (except for k, which expected at first position)
    :return: outer product of the eigenvector in the chosen energy (eigenvalue) window (unless vector = true)
    """
    evalues, evectors = sp.linalg.eigh(hamiltonian(k,**kwargs))

    ix = np.argwhere((evalues>eval_min) & (evalues < eval_max)) # Double Check
    band_vector = evectors.T[ix][0][0]

    if vector:
        return band_vector

    projector = np.outer(band_vector,np.conjugate(band_vector))

    return projector



def berry_phase_calculator(hamiltonian,eval_min,eval_max,**kwargs):
    #N = 100
    k_path = np.linspace(0.01,2*3.14, 20000)
    N = len(k_path)
    path_ordered_arrays = []
    for i,k in enumerate(k_path):
        if i == 0: # First state is just the vector
            path_ordered_arrays.append(specific_band_projector_calculator(hamiltonian,k,eval_min,eval_max,True,**kwargs))
        elif i == N-1: # Last state is just the vector (conjugated)
            path_ordered_arrays.append(np.conjugate(specific_band_projector_calculator(hamiltonian,k_path[0],eval_min,eval_max,True,**kwargs)))
        else: # Projectors
            path_ordered_arrays.append(specific_band_projector_calculator(hamiltonian,k,eval_min,eval_max,False,**kwargs))

    propagating_vector = path_ordered_arrays[0]

    for element in path_ordered_arrays[1:]:
        propagating_vector = np.matmul(element,propagating_vector)

    #return np.angle(propagating_vector) # returns the berry phase
    return -1j/(2*np.pi)*np.log(propagating_vector) # returns the polarization
    #return propagating_vector



if __name__ == "__main__":

    print('Projector Result: ',specific_band_projector_calculator(bulk_hamiltonian,0.06346651825433926,2,10,vector=False))
    print('Berry-Phase of 2xLC (Negative band): ',berry_phase_calculator(bulk_hamiltonian,-10,-2))
    print('Berry-Phase of 2xLC (Middle band): ',berry_phase_calculator(bulk_hamiltonian,-2,2))
    print('Berry-Phase of 2xLC (Positive band): ',berry_phase_calculator(bulk_hamiltonian,2,10))

    print('-----------------------------------------------')
    print('Weak Bonding Phase of 2xLC (for strong bond)')
    print('Berry-Phase of 2xLC (Negative band): ',berry_phase_calculator(bulk_hamiltonian,-5,-1,J1=1,J2=0.1))
    print('Berry-Phase of 2xLC (Middle band): ',berry_phase_calculator(bulk_hamiltonian,-2,2,J1=1,J2=0.1))
    print('Berry-Phase of 2xLC (Positive band): ',berry_phase_calculator(bulk_hamiltonian,1,5,J1=1,J2=0.1))

    print('-----------------------------------------------')
    # print('Trivial Phase')
    # w = 0
    # v = 1
    # print('Berry-Phase of positive band, SSH:  ', berry_phase_calculator(ssh_hamiltonian, 0.5, 3.5,w=w,v=v))
    # print('Berry-Phase of negative band, SSH:  ', berry_phase_calculator(ssh_hamiltonian, -3.5, -0.2,w=w,v=v))
    # print('-----------------------------------------------')
    #
    #
    # print('Topological Phase')
    # w = 1
    # v = 0
    # print('Berry-Phase of positive band, SSH:  ', berry_phase_calculator(ssh_hamiltonian, 0.5, 3.5,w=w,v=v))
    # print('Berry-Phase of negative band, SSH:  ', berry_phase_calculator(ssh_hamiltonian, -3.5, -0.2,w=w,v=v))
    # print('-----------------------------------------------')
    # Plot Spectra
    fig, ax_list = plt.subplots(2, 1, sharex=True)
    fig.figsize = (10, 15)

    J3_list = [0.44,0.56]
    fig.suptitle('Bulk Spectrum of 2xLC-Tree, Case 1')

    print('Berry-Phase of 2xLC (Negative band): ', berry_phase_calculator(bulk_hamiltonian, -1.2, -0.8, J1=1, J3=0.5))

    for i, J2 in enumerate(J3_list):
        spectrum = spectrum_calculator(bulk_hamiltonian,J1=1,J3=J2)

        ax_list[i].set_title('Strong Bond = ' + str(J2))
        ax_list[i].hist(spectrum, bins=200)
        ax_list[i].set_ylabel('Counts')

    ax_list[-1].set_xlabel('E/t')



    # k_list = [0,np.pi/4,np.pi/2,np.pi,7*np.pi/4]
    # for k in k_list:
    #     print('-----------  k = ' + str(k) + ' ------------------')
    #     print('Numerical Projector \n', specific_band_projector_calculator(ssh_hamiltonian,k,0.5,3.5,w=1,v=0))
    #     print('Actual Projector:\n',ssh_positive_band_projector(k))
    #
    # print('----------------------------------')
    # print('Negative band?')
    # for k in k_list:
    #     print('-----------  k = ' + str(k) + ' ------------------')
    #     print('Numerical Projector \n', specific_band_projector_calculator(ssh_hamiltonian,k,-3.5,-0.5,w=1,v=0))
    #     print('Actual Projector:\n',ssh_negative_band_projector(k))
    # w_list = [2,0.5]
    # fig.suptitle('Bulk Spectrum of SSH Model')
    #
    # for i, w in enumerate(w_list):
    #     spectrum = spectrum_calculator(ssh_hamiltonian,w=w,v=1)
    #
    #     ax_list[i].set_title('W = ' + str(w))
    #     ax_list[i].hist(spectrum, bins=200)
    #     ax_list[i].set_ylabel('Counts')
    #
    # ax_list[-1].set_xlabel('E/t')
    #
    # plt.tight_layout()
    plt.show()
