from abc import abstractmethod
from AbstractTree import AbstractTree
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiebCayley(AbstractTree):
    def __init__(self,M,k,save_ram=False):
        self.M = M
        self.mc = math.floor(M/2) # Number of Cayley Shells
        self.ml = math.ceil(M/2) # Number of Lieb Shells
        self.k = k # Connectivity (Degree of each node = k + 1)
        self.save_ram = save_ram
        logger.info('Initiating LiebCayley Tree')
        self.N = int(1 + ((k + 1)/(k-1))*(k**self.mc + k**self.ml -2)) # Number of Nodes
        if not save_ram:
            self.G = AbstractTree._tree_creator(self.N,self.k,LiebCayley._tree_edges) # networkx graph object
            self.A = nx.adjacency_matrix(self.G) # Sparse Matrix


    @staticmethod
    def _tree_edges(n,r):
        """
        Iteratively defines the tree structure
        :param n: Number of nodes
        :param r: Connectivity (k)
        :return: A list of tuples that define the edges of our tree
        """
        if n == 0:
            return
        # helper function for trees
        # yields edges in rooted tree at 0 with n nodes and branching ratio r
        nodes = iter(range(n))
        cayley_shell = [next(nodes)]  # stack of max length r
        lieb_shell = []
        r = r + 1
        first_run = True
        # Iterative filling of shells
        while cayley_shell:
            source = cayley_shell.pop(0)
            for i in range(r):
                try:
                    target = next(nodes)
                    lieb_shell.append(target)
                    yield source, target
                except StopIteration:
                    break
            if first_run:
                r = r - 1
                first_run = False

            if not cayley_shell:
                for lieb_node in lieb_shell:
                    try:
                        target = next(nodes)
                        cayley_shell.append(target)
                        yield lieb_node,target
                    except StopIteration:
                        break
                lieb_shell = []


    def shell_list(self):
        """Returns a list of lists nodes, with each list containing the nodes of the same shell
        Makes use of the fact that the construction algorithm constructs each shell after the other

        Returns
        -------
        shell_lists : Multiple lists corresponding to the number of Shells M + 1 (incl. the center)
            each list containing the nodes of that shell
        """
        nodes = list(self.G.nodes)
        shell_lists = [[0]]
        l = 1
        for shell_number in range(1, self.M + 1):
            n = (self.k + 1) * (self.k ** (math.ceil(shell_number/2) - 1))
            shell_lists.append(nodes[l:(l + n)])
            l += n
        return shell_lists


    def polynomial_recursion_relation(self,polys,l):
        if self.M % 2 == 0: #Tree ends on a lieb-shell
          pass


    def _eff_hamiltonian_constructor(self,l,J = None):
        """
        Constructs the effective Hamiltonian of the Lieb-Cayley tree for a given shell number.
        Automatically adapts to M even or M odd cases. For M even, the matrix will have dimension M - l + 1.
        :param l: Number of the shell we're starting our construction from. For l = 0 we start from |0>
        :return: 2D-nparray of effective hamiltonian h and scalar of degeneracy d of the hamiltonian
        """
        if not J and J != 0:
            J = 1
        a = [np.sqrt(self.k)*J,1]
        offdiag = np.tile(a,reps=int(np.ceil(self.M/2)))
        offdiag[0] = np.sqrt(self.k + 1)*J
        # print('Length Offdiagonal:', len(offdiag))

        h = np.diag(offdiag[l:self.M],k=1) + np.diag(offdiag[l:self.M],k=-1) #The M automatically cuts off the last element of the offdiagonal if M is odd
        # print('Shape of H: ',np.shape(h))
        # print(l,',h:',h,self.M%2)
        # print('-----------------------------------------------------------')
        return h

    def _eff_hamiltonian_list(self,J = None):
        """
        Returns a list of all effective hamiltonians and a second list with the degeneracies of the eigenvalues
        of these hamiltonians. The ordering of these to lists must be 1-to-1.
        :param J: Hopping parameter from Cayley Shells too outer Lieb shells (scale sqrt(k) values)
        :return:
        """

        Hs = []
        degeneracies = []
        # Symm States (once for l = 0 and l = 1)
        Hs.append(self._eff_hamiltonian_constructor(0,J))
        degeneracies.append(1)
        Hs.append(self._eff_hamiltonian_constructor(1,J))
        degeneracies.append(self.k) # Why is this not k+1?
        # Anti-Symm States (runs from 1 to mc with lc = 2l)
        for lc in range(1,self.mc + self.M%2):
            Hs.append(self._eff_hamiltonian_constructor(2*lc + 1,J))
            degeneracies.append((self.k -1)*(self.k + 1)*self.k**(lc-1))
        # print('degen:',degeneracies)
        return Hs, degeneracies



class DoubleLiebCayley(AbstractTree):
    def __init__(self,M,k,save_ram=False):
        self.M = M
        self.mc = math.floor(M/3) # Number of Cayley Shells
        self.ml1 = math.floor(M/3) + math.ceil(M%3 /2) # Number of Lieb-1 Shells
        self.ml2 = math.floor(M/3) + math.floor(M%3 /2)
        self.k = k # Connectivity (Degree of each node = k + 1)
        self.save_ram = save_ram

        self.N = int(1 + ((k + 1)/(k-1))*(k**self.mc + k**self.ml1 + k**self.ml2 -3)) # Number of Nodes
        if not save_ram:
            self.G = AbstractTree._tree_creator(self.N,self.k,DoubleLiebCayley._tree_edges) # networkx graph object
            self.A = nx.adjacency_matrix(self.G) # Sparse Matrix


    @staticmethod
    def _tree_edges(n,r):
        """
        Iteratively defines the tree structure
        :param n: Number of nodes
        :param r: Connectivity (k)
        :return: A list of tuples that define the edges of our tree
        """
        if n == 0:
            return
        # helper function for trees
        # yields edges in rooted tree at 0 with n nodes and branching ratio r
        nodes = iter(range(n))
        cayley_shell = [next(nodes)]  # stack of max length r
        lieb_shell1 = []
        lieb_shell2 = []
        r = r + 1
        first_run = True
        # Iterative filling of shells
        while cayley_shell:
            source = cayley_shell.pop(0)
            for i in range(r):
                try:
                    target = next(nodes)
                    lieb_shell1.append(target)
                    yield source, target
                except StopIteration:
                    break
            if first_run:
                r = r - 1
                first_run = False

            if not cayley_shell:
                for lieb_node in lieb_shell1:
                    try:
                        target = next(nodes)
                        lieb_shell2.append(target)
                        yield lieb_node,target
                    except StopIteration:
                        break
                lieb_shell1 = []

            if not cayley_shell:
                for lieb_node in lieb_shell2:
                    try:
                        target = next(nodes)
                        cayley_shell.append(target)
                        yield lieb_node,target
                    except StopIteration:
                        break
                lieb_shell2 = []


    def shell_list(self):
        """Returns a list of lists nodes, with each list containing the nodes of the same shell
        Makes use of the fact that the construction algorithm constructs each shell after the other

        Returns
        -------
        shell_lists : Multiple lists corresponding to the number of Shells M + 1 (incl. the center)
            each list containing the nodes of that shell
        """
        nodes = list(self.G.nodes)
        shell_lists = [[0]]
        l = 1
        for shell_number in range(1, self.M + 1):
            n = (self.k + 1) * (self.k ** (math.ceil(shell_number/3) - 1))
            shell_lists.append(nodes[l:(l + n)])
            l += n
        return shell_lists


    def polynomial_recursion_relation(self,polys,l):
        if self.M % 2 == 0: #Tree ends on a lieb-shell
          pass


    def _eff_hamiltonian_constructor(self,l,J = None):
        """
        Constructs the effective Hamiltonian of the Lieb-Cayley tree for a given shell number.
        Automatically adapts to M even or M odd cases. For M even, the matrix will have dimension M - l + 1.
        :param l: Number of the shell we're starting our construction from. For l = 0 we start from |0>
        :return: 2D-nparray of effective hamiltonian h and scalar of degeneracy d of the hamiltonian
        """
        if not J and J != 0:
            J = 1
        a = [np.sqrt(self.k)*J,1,1]
        offdiag = np.tile(a,reps=int(np.ceil(self.M/3)))
        offdiag[0] = np.sqrt(self.k + 1)*J
        # print('Length Offdiagonal:', len(offdiag))

        h = np.diag(offdiag[l:self.M],k=1) + np.diag(offdiag[l:self.M],k=-1) #The M automatically cuts off the last element of the offdiagonal if M is odd
        # print('Shape of H: ',np.shape(h))
        # print(l,',h:',h,self.M%3)
        # print('-----------------------------------------------------------')
        return h

    def _eff_hamiltonian_list(self,J = None):
        """
        Returns a list of all effective hamiltonians and a second list with the degeneracies of the eigenvalues
        of these hamiltonians. The ordering of these to lists must be 1-to-1.
        :param J: Hopping parameter from Cayley Shells too outer Lieb shells (scale sqrt(k) values)
        :return:
        """

        Hs = []
        degeneracies = []
        # Symm States (once for l = 0 and l = 1)
        Hs.append(self._eff_hamiltonian_constructor(0,J))
        degeneracies.append(1)
        Hs.append(self._eff_hamiltonian_constructor(1,J))
        degeneracies.append(self.k) # Why is this not k+1?
        # Anti-Symm States (runs from 1 to mc with lc = 2l)
        for lc in range(1,self.mc + math.ceil(self.M%3 / 2)):
            Hs.append(self._eff_hamiltonian_constructor(3*lc + 1,J))
            degeneracies.append((self.k -1)*(self.k + 1)*self.k**(lc-1))
        # print('degen:',degeneracies)
        return Hs, degeneracies




if __name__ == "__main__":
    C1 = LiebCayley(5,2)
    C2 = DoubleLiebCayley(6,2)
    fig, ax_list = plt.subplots(2,1,sharex=True)
    fig.figsize = (15,10)

    eval, evec = C2.exact_diagonalization()
    ax_list[0].hist(eval,bins=100)
    ax_list[0].set_ylabel('D')
    ax_list[0].set_xlabel('E/t')
    ax_list[0].set_title('Exact Diagonalization Spectrum')

    # C1.draw(ax_list[1])
    # C2.draw(ax_list[0])

    eval2 = C2.effective_diagonalization()
    ax_list[1].hist(eval2,bins=100)
    ax_list[1].set_ylabel('D')
    ax_list[1].set_xlabel('E/t')
    ax_list[1].set_title('Effective Hamiltonian Diagonalization Spectrum')

    # eval, evec = C1.exact_diagonalization()
    # ax_list[1].hist(eval,bins=100)
    # ax_list[1].set_ylabel('D')
    # ax_list[1].set_xlabel('E/t')
    # ax_list[1].set_title('Exact Diagonalization Spectrum')

    plt.show()