from unittest.mock import right

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

def bulk_hamiltonian(k,J1=1,J2=1):
    """
    Bulk hamiltonian of the 2xLieb-Cayley Tree (UC of the Case 1 Tree)
    Assumes that connectivity K = 16
    :param k: crystal momentum to be iterated over
    :param J1: 2nd weak-bond coefficient
    :param J2: strong-bond coefficient
    :return: returns the numerical form of the bulk hamiltonian for the given values
    """
    return np.array([[0,1,J2*4*np.exp(-1j*k)],
                     [1,0,J1*1],
                     [J2*4*np.exp(1j*k),J1*1,0]])



def ssh_hamiltonian(k,w=2,v=1): # Top. non trivial phase
    return np.array([[0, v + w*np.exp(1j*k)],
                     [v + w*np.exp(-1j*k),0]])



def spectrum_calculator(hamiltonian,**kwargs):
    """
    Numerically calculates the spectrum for a given bulk hamiltonian, by scanning thorugh k-space [0,2pi] and calculating
    eigenvalues for each k-value.
    :param hamiltonian: A bulk hamiltonian that depends on k(!)
    :param *args: Allows to pass any other arguments that are fed into the hamiltonian (positionally)
    :return: The discretized spectrum of the bulk hamiltonian
    """
    spectrum = []
    for k in np.linspace(0,2*np.pi,1000):
        evalues = sp.linalg.eigh(hamiltonian(k,**kwargs),eigvals_only=True)
        spectrum.extend(evalues.tolist())
    return spectrum


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
    # Normalize to 1?
    # print('Length of Band vector: ',np.linalg.norm(band_vector))

    if vector:
        return band_vector

    projector = np.outer(band_vector, np.conjugate(band_vector))

    return projector



def berry_phase_calculator(hamiltonian,eval_min,eval_max,**kwargs):
    #N = 100
    k_path = np.arange(0, 2 * np.pi, 0.0001)
    N = len(k_path)
    path_ordered_arrays = []
    for i,k in enumerate(k_path):
        #print(k)
        if i == 0: # First state is just the vector
            path_ordered_arrays.append(specific_band_projector_calculator(hamiltonian,k,eval_min,eval_max,True,**kwargs))
        elif i == N-1: # Last state is just the vector (conjugated)
            path_ordered_arrays.append(np.conjugate(specific_band_projector_calculator(hamiltonian,k,eval_min,eval_max,True,**kwargs)))
        else: # Projectors
            path_ordered_arrays.append(specific_band_projector_calculator(hamiltonian,k,eval_min,eval_max,False,**kwargs))

    propagating_vector = path_ordered_arrays[0]
    # print('Propagting vector',propagating_vector)
    # print('2nd element PO-array',path_ordered_arrays[1])
    # print('-------------------------------------------------')

    for element in path_ordered_arrays[1:]:
        propagating_vector = np.matmul(element,propagating_vector)
        print(propagating_vector)

    #return np.angle(propagating_vector) # returns the berry phase
    return -1j/(2*np.pi)*np.log(propagating_vector) # returns the polarization



if __name__ == "__main__":

    #print('Projector Result: ',specific_band_projector_calculator(bulk_hamiltonian,0.06346651825433926,2,10,vector=False))
    print('-----------------------------------------------')
    print('Berry-Phase of 2xLC (Negative band): ',berry_phase_calculator(bulk_hamiltonian,-10,-2))
    print('Berry-Phase of 2xLC (Middle band): ',berry_phase_calculator(bulk_hamiltonian,-2,2))
    print('Berry-Phase of 2xLC (Positive band): ',berry_phase_calculator(bulk_hamiltonian,2,10))
    print('-----------------------------------------------')
    w = 0
    v = 1
    print('Berry-Phase of positive band, SSH:  ', berry_phase_calculator(ssh_hamiltonian, 0.5, 3.5,w=w,v=v))
    print('Berry-Phase of negative band, SSH:  ', berry_phase_calculator(ssh_hamiltonian, -3.5, -0.2,w=w,v=v))
    print('-----------------------------------------------')
    # Plot Spectra
    fig, ax_list = plt.subplots(2, 1, sharex=True)
    fig.figsize = (10, 15)

    J2_list = [1,0.25]
    fig.suptitle('Bulk Spectrum of 2xLC-Tree, Case 1')

    for i, J2 in enumerate(J2_list):
        spectrum = spectrum_calculator(bulk_hamiltonian,J1=1,J2=J2)

        ax_list[i].set_title('Strong Bond = ' + str(J2))
        ax_list[i].hist(spectrum, bins=200)
        ax_list[i].set_ylabel('Counts')

    ax_list[-1].set_xlabel('E/t')

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
