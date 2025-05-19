from abc import abstractmethod

from AbstractTree import AnisotropicAbstractTree
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import math
import logging
import itertools

class HaldaneCayleyTree(AnisotropicAbstractTree):
    def __init__(self,M,t2=1,save_ram=False):
        """
        :param k: int, Number of children per node (k = r + 1)
        :param M: int, Number of shells of the Cayley Tree, must be larger than 1 due to construction algorithm
        :param t2: float, Scaling of the complex hopping amplitude.
        """
        self.k = 2
        self.M = M
        self.N = 1 + (self.k+1)*(2**M -1) # Number of nodes on the Haldane-cayley tree (== N Cayley Tree)
        self.save_ram = save_ram
        self.t2 = t2

        if not save_ram:
            self.G = nx.empty_graph(self.N, nx.DiGraph)
            self.G.add_edges_from(HaldaneCayleyTree._tree_edges(self.N, self.k)) # networkx graph object

            # Now we construct the next-nearest neighbour hopping
            # First construct the lower-shell nn hoping
            for node in self.G.nodes:
                nextnearestneighbours = HaldaneCayleyTree.sub_nnneighbours(self.G,node)
                for nnnode in nextnearestneighbours:
                    if nnnode %2 == 0:
                        self.G.add_edge(node,nnnode,weight=-1j * self.t2)
                        self.G.add_edge(nnnode, node, weight=1j * self.t2)
                    elif nnnode %2 == 1:
                        self.G.add_edge(node,nnnode,weight=1j * self.t2)
                        self.G.add_edge(nnnode, node, weight=-1j * self.t2)
                    else: # If there is no node anymore, we break out
                        break
            for node in self.G.nodes:
                if node % 2 == 0 and node > 3:
                    self.G.add_edge(node,node+1,weight=-1j * self.t2)
                    self.G.add_edge(node+1, node, weight=1j * self.t2)
                else:
                    continue

            # Manual add the inner circle
            self.G.add_edge(2, 1,weight=1j * self.t2)
            self.G.add_edge(1, 2, weight=-1j * self.t2)

            self.G.add_edge(1, 3, weight=1j * self.t2)
            self.G.add_edge(3, 1, weight=-1j * self.t2)

            self.G.add_edge(3, 2, weight=1j * self.t2)
            self.G.add_edge(2, 3, weight=-1j * self.t2)

    @staticmethod
    def _tree_edges(n, r):
        if n == 0:
            return
        # helper function for trees
        # yields edges in rooted tree at 0 with n nodes and branching ratio r
        nodes = iter(range(n))
        parents = [next(nodes)]  # stack of max length r

        first_time = True
        r = r + 1

        while parents:
            source = parents.pop(0)
            for i in range(r):
                try:
                    target = next(nodes)
                    parents.append(target)
                    yield source, target, {'weight': 1}
                    yield target, source, {'weight': 1}
                except StopIteration:
                    break
            if first_time:
                r = r - 1
            first_time = False

    @staticmethod
    def sub_nnneighbours(G, chosen_node, flatten=False):
        """
        For a chosen node, this returns all the nodes that follow in the tree (called subnodes)
        which are next-nearest neighbours to the chosen nodes.
        :param G: networkx.graph, The graph object
        :param chosen_node: int, The node of which you want to find the subtree
        :param flatten: Boolean, If true the returned array will not be flat instead of grouped according to
                                parent node
        :return: A list of all nodes that form the subtree.
        """
        node_list = []
        work_list = [n for n in G.neighbors(chosen_node) if
                     n > chosen_node]  # all nearest neighbours to node n in lower shell
        while work_list != []:
            nnumber = work_list.pop(0)
            subnodes = [n for n in G.neighbors(nnumber) if n > nnumber]
            node_list.extend(subnodes)

        if flatten:
            node_list = [item for row in node_list for item in row]

        return node_list

    @property
    def A(self):
        A = self.to_numpy_complex()
        return A

    def shell_list(self):
        """Returns a list of lists nodes, with each list containing the nodes of the same shell
            Makes use of the fact that the construction algorithm constructs each shell after the other

            Returns
            -------
            shell_lists : Multiple lists corresponding to the number of Shells M
                each list containing the nodes of that shell
            """
        nodes = list(self.G.nodes)
        shell_lists = [[0]]  # already contains the 0th node
        l = 1  # position of last added node
        for shell_number in range(1, self.M + 1):
            n = (self.k + 1) * (self.k ** (shell_number - 1))
            shell_lists.append(nodes[l:(l + n)])
            l += n

        return shell_lists

    def draw(self,ax):

        # Drawing the model
        pos = self.shell_layout(H.G, H.shell_list())
        nx.draw_networkx_nodes(H.G, pos, node_size=10, ax=ax)

        edge_undirected = [(u, v) for (u, v, d) in H.G.edges(data=True) if d["weight"] == 1]

        # We only draw one direction of the complex edges
        edge_directed = [(u, v) for (u, v, d) in H.G.edges(data=True) if d["weight"] == self.t2*1j]

        # Turn off arrows because they are technically undirected
        nx.draw_networkx_edges(H.G, pos, edgelist=edge_undirected, width=1, arrows=False, ax=ax)

        nx.draw_networkx_edges(
            H.G, pos, edgelist=edge_directed, width=1, alpha=0.5, edge_color="violet", style="dashed", ax=ax)


if __name__ == "__main__":
    H = HaldaneCayleyTree(4,1)

    fig, ax_list = plt.subplots(2,1)
    fig.figsize = (15,10)

    eval, evec = H.exact_diagonalization()
    ax_list[0].hist(eval,bins=200)
    ax_list[0].set_ylabel('D')
    ax_list[0].set_xlabel('E/t')
    ax_list[0].set_title('Exact Diagonalization Spectrum with M = ' + str(H.M) + ' and N = ' + str(H.N))

    # Drawing the model
    H.draw(ax_list[1])
    plt.tight_layout()
    plt.show()