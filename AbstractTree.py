from abc import ABC,abstractmethod

import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import math

from PIL.ImImagePlugin import number


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

    @abstractmethod
    def shell_list(self) -> None:
        pass

    @abstractmethod
    def polynomial_recursion_relation(self,polys,l):
        pass

    @abstractmethod
    def _eff_hamiltonian_list(self):
        pass

    @staticmethod
    def _tree_creator(n,r,_tree_edges,create_using=nx.Graph):
        G = nx.empty_graph(n, create_using)
        G.add_edges_from(_tree_edges(n, r))
        return G

    def exact_diagonalization(self, eigvals_only=False, noise=None):
        """
        Performs exact diagonalization on the adjacency matrix of the Tree.
        Transforms Adjacency matrix to a dense expression, thereby limiting the max size that can be solved.
        If Noise is not None, adds on-site disorder by adding a diagonal to the Hamiltonian whose elements are drawn from
        a uniform distribution with min = 0 and max = 1 multiplied by noise. (thereby noise gives the maximal value for the noise).


        :param noise: Default is None. Should be a order of magnitude, gives max size of noise applied to diagonal of Hamiltonian
        :param eigvals_only: Choose True if you only want the eigenvalues of the graph
        :return: w: array of N eigenvalues
        :return: v: array representing the N eigenvectors
        """
        # Note: We turn A into a dense matrix here -> limiting factor
        A = self.A.todense()
        if noise:
            random_generator = np.random.default_rng()
            random_noise = random_generator.uniform(size=self.N)*noise
            A += np.diag(random_noise)

        return sp.linalg.eigh(A,eigvals_only=eigvals_only) #Assumes symmetric matrix -> Might not be the case in the future

    def effective_diagonalization(self):
        """
        Diagonalizes the effective Hamiltonians which define the dynamics of the symmetrized states on the tree.
        Returns a list of eigenvalues and a list of weights assosciated with each of these eigenvalues.
        :return: eigenval: The eigenvalues of you tree determined form the effective hamiltonians. Rounded to 10^{-5}
        weights: the weight of each eigenvalue determined from the degeneracy of the associated hamiltonian
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

    def draw(self,ax):
        """
        :param ax: The axis on which you want to plot your cayley tree
        :param color_shells: If True, instead of the branches, the shells will have the same color
        :return: None
        """
        nlist = self.shell_list()
        # print(nlist)
        # blist = tc.branch_list(self.G,self._r,self._M)
        # print(blist)
        clist = self._color_list(nlist)

        # if color_shells:
        #     clist = tc.color_list(self._r, self._M, nlist)

        nx.draw(self.G,pos=self.shell_layout(self.G,nlist=nlist),ax=ax,node_shape='.',node_color=clist,cmap='tab20')


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
            subnodes = [n for n in self.G.neighbors(nnumber) if n > nnumber]
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
            radius = radius_bump


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