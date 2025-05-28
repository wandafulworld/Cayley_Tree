from abc import ABC,abstractmethod
from logging import raiseExceptions

import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import math
import logging

from PIL.ImImagePlugin import number
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Abstract Tree Class
class AbstractTree(ABC):
    def __init__(self):
        # Need to be overwritten by subclass
        self.G = None
        self.k = None
        self.M = None
        self.N = None
        self.A = None
        self.save_ram = False
        logger.info('Initiating Abstract Tree')

    @abstractmethod
    def shell_list(self) -> None:
        pass

    def exact_diagonalization(self, eigvals_only=False, on_site_noise=None, bond_noise=None, random_seed = None):
        """
        Performs exact diagonalization on the adjacency matrix of the tree. (Which is the Hamiltonian in a TB-model).
        Uses Adjacency matrix in a dense expression, thereby limiting the max size that can be solved.
        If on_site_noise is not None, adds on-site disorder by adding a diagonal to the Hamiltonian whose elements are drawn from
        a uniform distribution with min = 0 and max = 1 multiplied by noise. (thereby noise gives the maximal value for the noise).
        If bond_noise it not None, adds noise drawn from a uniform distribution with bounds given by +/- bond_noise to each off-diagonal element
        in the adjacency matrix.

        :param on_site_noise: float, Default is None. Should be a order of magnitude, gives max size of noise applied to diagonal of Hamiltonian
        :param bond_noise: float, Default is None. Gives max size of noise applied to the non-diagonal element of the Hamiltonian (hopping bonds)
        :param eigvals_only: boolean, Choose True if you only want the eigenvalues of the graph
        :param random_seed: int, optional, Seed for reproducibility.
        :return: w: array of N eigenvalues
        :return: v: array representing the N eigenvectors
        """
        A = self.A # Local such that noise is not added permanently
        n = A.shape[0]
        if on_site_noise:
            random_generator = np.random.default_rng(seed=random_seed)
            random_noise = random_generator.uniform(-1,1,size=self.N)*on_site_noise
            A = A + np.diag(random_noise)

        if bond_noise: #ToDo: Check compatibility with complex A
            random_generator = np.random.default_rng(seed=random_seed)
            # mask of upper-triangle positions where A is nonzero
            upper_mask = (np.triu(A, k=1) != 0)
            # draw noise only for those positions
            noise_upper = np.zeros((n, n))
            # uniform in [-bond_noise, +bond_noise] where mask is True
            noise_vals = random_generator.uniform(-bond_noise, bond_noise, size=upper_mask.sum())
            noise_upper[np.triu_indices(n, k=1)[0][upper_mask[np.triu_indices(n, k=1)]],
                        np.triu_indices(n, k=1)[1][upper_mask[np.triu_indices(n, k=1)]]] = noise_vals
            # symmetrize
            noise = noise_upper + noise_upper.T
            # add to A
            A = A + noise

        # Calculate Eigenvalues / Vectors
        if A.shape[1] == n and np.allclose(A, A.T): # If A symmetric use eigh
            return sp.linalg.eigh(A,eigvals_only=eigvals_only)
        else:
            if eigvals_only:
                return sp.linalg.eigvals(A)
            else:
                return sp.linalg.eig(A)


    def _color_list(self,nlists):
        """Returns a list of color gradients (0,1) where the list positions correspond to node position given through the nlists.
            Each node list in nlists will get a different color gradient that can then be mapped using a cmap

            Parameters
            ----------
            nlists : A list of lists of nodes of a Graph. Each sublist will get a color assigned.

            Returns
            -------
            color_list : One list of color positions (between 0 and 1), where all the nodes of the same sublist will have the same color position
                This lists then needs to be mapped to a colormap via the draw function of networkx
                The 0th node will always be set to 0
            """
        # Determine number of colors needed
        n_colors = len(nlists)
        color_pos = np.linspace(0, 1, n_colors + 1)

        color_array = np.zeros(self.N)
        for i, sublist in enumerate(nlists):
            np.put(color_array, ind=sublist, v=color_pos[i + 1])

        return color_array.tolist()

    def _subbrancher(self,chosen_node, flatten=False):
        """
        For a chosen node, this returns all the nodes that follow in the tree (called subnodes).
        Useful for constructing the anti-symmetric states of a tree.
        :param G: The graph object
        :param chosen_node: The node of which you want to find the subtree
        :param flatten: If True, flattens the node list
        :return: A list of all nodes that form the subtree.
        """
        node_list = []
        work_list = [chosen_node]
        while work_list:
            nnumber = work_list.pop(0)
            subnodes = [n for n in self.G.neighbors(nnumber) if n > nnumber + self.k]
            work_list.extend(subnodes)
            if subnodes:
                node_list.append(subnodes)

        if flatten:
            node_list = [item for row in node_list for item in row]

        return node_list

    def _linear_combination_vector(self,nodes, prefactors, normalize=False):
        """
        Returns a numpy array that is a linear combination of the nodes provided via "nodes" in position basis
        "nodes" can be a list of lists, where each list corresponds to scaling prefactor at the equivalent position in "prefactors"
        :param nodes: list of Nodes that make the vector, number of sublists needs to correspond to number of prefactors, order needs to be the same
        :param prefactors: List of prefactor for each sublist in the "nodes" parameter, allows for scaled linear combinations
        :param normalize: If True, the vector will be normalized to 1
        :return: 1D array with length N
        """
        vector = np.zeros(self.N, dtype=complex)

        if len(prefactors) == 1:
            vector[nodes] = 1 * prefactors[0]
        else:
            for i, prefactor in enumerate(prefactors):
                vector[nodes[i]] = prefactor

        if normalize:
            vector = vector / np.sqrt(np.sum(np.square(vector)))

        return vector

    def shell_layout(self,G, nlist=None, scale=1, center=None, dim=2):
        """Position nodes in concentric circles.

        Parameters
        ----------
        G : NetworkX graph or list of nodes
            A position will be assigned to every node in G.

        nlist : list of lists
           List of node lists for each shell.

        rotate : angle in radians (default=pi/len(nlist))
           Angle by which to rotate the starting position of each shell
           relative to the starting position of the previous shell.
           To recreate behavior before v2.5 use rotate=0.

        scale : number (default: 1)
            Scale factor for positions.

        center : array-like or None
            Coordinate pair around which to center the layout.

        dim : int
            Dimension of layout, currently only dim=2 is supported.
            Other dimension values result in a ValueError.

        Returns
        -------
        pos : dict
            A dictionary of positions keyed by node

        Raises
        ------
        ValueError
            If dim != 2
        """
        if dim != 2:
            raise ValueError("can only handle 2 dimensions")

        if center is None:
            center = np.zeros(dim)
        else:
            center = np.asarray(center)

        if len(G) == 0:
            return {}
        if len(G) == 1:
            return {nx.utils.arbitrary_element(G): center}

        if nlist is None:
            # draw the whole graph in one shell
            nlist = [list(G)]

        radius_bump = scale / len(nlist)

        if len(nlist[0]) == 1:
            # single node at center
            radius = 0.0
        else:
            # else start at r=1
            radius = radius_bump*0.5


        sectors = np.linspace(0,2 * np.pi, self.k + 2,dtype=np.float32)
        npos = {}
        for nodes in nlist:
            number_per_sector = int(len(nodes)/(self.k + 1))
            if len(nodes) == 1:
                number_per_sector = 1

            theta = []
            for i in range(len(sectors) - 1):
                delta_theta = (sectors[i+1] - sectors[i])/(number_per_sector + 1)
                for j in range(1,number_per_sector+1):
                    theta.append(sectors[i] + j*delta_theta)
            theta = np.array(theta)

            pos = radius * np.column_stack([np.cos(theta), np.sin(theta)]) + center
            npos.update(zip(nodes, pos))
            radius += radius_bump

        return npos

    def nth_root_of_unity(n, k, precise=False):
        """
        Returns the k-root of the nth-root of unity
        :param n: Determines the root degree of unity
        :param k: Choose which root of the n roots you want
        :param precise: If False, returns root rounded to 5th decimal, else to numeric precision
        :return: complex number
        """
        if precise:
            return np.exp((2 * k * np.pi * 1j) / n)
        else:
            return np.round(np.exp((2 * k * np.pi * 1j) / n), 5)


class IsotropicAbstractTree(AbstractTree):

    @abstractmethod
    def _eff_hamiltonian_list(self):
        pass

    @staticmethod
    def _tree_creator(n,r,_tree_edges,create_using=nx.Graph,**kwargs):
        G = nx.empty_graph(n, create_using)
        G.add_edges_from(_tree_edges(n, r,**kwargs))
        return G

    def mahan_diagonalization(self):
        """
        Using Mahans approach, diagonalizes the effective Hamiltonians which define the dynamics of the symmetry sectors.
        Returns a list of eigenvalues and a list of weights assosciated with each of these eigenvalues.
        :return: eigenval: The eigenvalues of you tree determined form the effective hamiltonians. Rounded to 10^{-5}
        :return: weights: the weight of each eigenvalue determined from the degeneracy of the associated hamiltonian
        """
        hs, degeneracies = self._eff_hamiltonian_list()
        eigenval = []
        weights = []
        for i,h in enumerate(hs):
            eval = sp.linalg.eigh(h,eigvals_only=True)
            eval = np.round(eval,5)
            # print(eval)
            if not self.save_ram:
                eigenval.extend(np.repeat(eval,degeneracies[i]).tolist())
            else:
                eigenval.extend(eval)
                weights.extend(np.repeat(degeneracies[i], len(eval)).tolist())

        if not self.save_ram:
            return eigenval
        else:
            return eigenval, weights

    def effective_diagonalization(self):
        """ For backward compatibility """
        return self.mahan_diagonalization()


    def draw(self,ax):
        """
        :param ax: The axis on which you want to plot your tree
        :return: None
        """
        nlist = self.shell_list()
        clist = self._color_list(nlist)
        nx.draw(self.G,pos=self.shell_layout(self.G,nlist=nlist),ax=ax,node_shape='.',node_color=clist,cmap='tab20')


class AnisotropicAbstractTree(AbstractTree):

    def to_numpy_complex(self):
        """Turns the Graph G into a np.complex128 adjacency matrix, drawing edge weights from graph data.
         Allows for complex edge weights."""
        N_size = len(self.G.nodes())
        E = np.zeros(shape=(N_size, N_size), dtype=np.complex128)

        for i, j, attr in self.G.edges(data=True):
            E[i, j] = attr.get("weight")
        return E

    def mahan_diagonalization(self):
        raise Exception("Anisotropic Trees can not be block-diagonalized using Mahans approach.")

    @abstractmethod
    def draw(self,ax):
        pass
