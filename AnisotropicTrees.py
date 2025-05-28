"""
Cayley tree inspired models that are anisotropic, i.e. edges are directed.
This breaks the permutation symmetry of the tree and therefore does not permit using Mahans approach.
"""

from AbstractTree import AnisotropicAbstractTree
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import QWZ_HelperFunctions as hf

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
        """
        Draws the Graph on the axis provided
        :param ax: matplotlib.pyplot.axis object, axis on which the graph should be drawn
        :return: None
        """

        # Drawing the model
        pos = self.shell_layout(self.G, self.shell_list())
        nx.draw_networkx_nodes(self.G, pos, node_size=10, ax=ax)

        edge_undirected = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] == 1]

        # We only draw one direction of the complex edges
        edge_directed = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] == self.t2*1j]

        # Turn off arrows because they are technically undirected
        nx.draw_networkx_edges(self.G, pos, edgelist=edge_undirected, width=1, arrows=False, ax=ax)

        nx.draw_networkx_edges(
            self.G, pos, edgelist=edge_directed, width=1, alpha=0.5, edge_color="violet", style="dashed", ax=ax)



class QWZCayleyTree(AnisotropicAbstractTree):
    def __init__(self,M,t_sigma=1,scale=0.5,m=1,save_ram=False):
        """
        :param M: Number of shells of the QWZ-Cayley Tree, must be larger than 1 due to construction algorithm
        :param t_sigma: Scaling of the off-diagonal hopping (mixing)
        :param scale: Scaling of all hopping matrices (usually 0.5 for QWZ)
        :param m: mass term
        :param save_ram: If true, initializes tree with parameters only, no adjacency matrix / graph object
        """
        if M < 1:
            raise ValueError("M has to be at least 1")
        self.k = 3 # Degree = k + 1
        self.M = M
        self.N = int(2*(1 + (self.k+1)*(self.k**M -1)/(self.k-1))) # Number of nodes on the QWZ-cayley tree (== 2xN Cayley Tree)
        self.save_ram = save_ram

        self.t_sigma = t_sigma
        self.scale = scale
        self.m = m

        if not save_ram:
            self.G = nx.empty_graph(self.N, nx.DiGraph)
            hf.qwz_inner_ring(self.G,[0,1],[2,3,4,5,6,7,8,9],t_sigma=self.t_sigma,scale=self.scale) # networkx graph object
            if M > 1:
                parents = [2, 3, 4, 5, 6, 7,8,9]
                n = 10 # position of last added node
                for shell_number in range(2,M+1):
                    shell_size = 2*((self.k + 1) * (self.k ** (shell_number - 1)))
                    nodes = list(range(n,n+shell_size))
                    #print(nodes)

                    while parents:
                        local_parents = [parents.pop(0),parents.pop(0)]
                        next_nodes = [nodes.pop(0),nodes.pop(0),nodes.pop(0),nodes.pop(0),nodes.pop(0),nodes.pop(0)]
                        hf.qwz_connection_builder(self.G,local_parents,next_nodes,
                                                  self.G._node[local_parents[0]]['location'],
                                                  t_sigma=self.t_sigma,scale=self.scale)

                    parents = list(range(n,n+shell_size))
                    n += shell_size

    @property
    def A(self):
        A = self.to_numpy_complex()
        A = A + np.diag(np.tile([self.m, -self.m], int(self.N/2))) #Add On-Site Potential
        return A

    def shell_list(self):
        """Returns a list of lists, with each list containing the nodes of the same shell
            Makes use of the fact that the construction algorithm constructs each shell after the other

            Returns
            -------
            shell_lists : Multiple lists corresponding to the number of Shells M
                each list containing the nodes of that shell
            """
        nodes = list(self.G.nodes)
        shell_lists = [[0,1]]  # already contains the 0th and 1st node in the center
        l = 2  # position of last added node
        for shell_number in range(1, self.M + 1):
            n = 2*(self.k + 1) * (self.k ** (shell_number - 1))
            shell_lists.append(nodes[l:(l + n)])
            l += n

        return shell_lists

    def qwz_square_layout(self, nlist=None, center=None, dim=2):
        """Position nodes on square around the parent node, depending on relative position.

        Parameters
        ----------
        nlist : list of lists
           List of node lists for each shell.

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

        if len(self.G) == 0:
            return {}
        if len(self.G) == 1:
            return {nx.utils.arbitrary_element(self.G): center}

        if nlist is None:
            # draw the whole graph in one shell
            nlist = [list(self.G)]

        radius = 0.3

        npos = {}
        for nodes in nlist:
            nodes_pos = np.zeros((len(nodes), 2))
            if len(nodes) == 2:
                pos = np.array([[0, -0.01], [0, 0.01]])
                npos.update(zip(nodes, pos))

            else:
                for i, node in enumerate(nodes):
                    parent = [n for n in self.G.neighbors(node) if n < node][0]
                    parent_position = npos[parent] + np.array([-0.001, -0.001])
                    node_location = self.G._node[node]['location']
                    if node_location == 'south':
                        node_pos = parent_position + np.array([-radius, 0])
                        if i % 2 == 1:
                            nodes_pos[i] = node_pos + np.array([-0.001, -0.001])
                        else:
                            nodes_pos[i] = node_pos + np.array([0.001, 0.001])

                    elif node_location == 'east':
                        node_pos = parent_position + np.array([0, radius])
                        if i % 2 == 1:
                            nodes_pos[i] = node_pos + np.array([-0.001, -0.001])
                        else:
                            nodes_pos[i] = node_pos + np.array([0.001, 0.001])

                    elif node_location == 'north':
                        node_pos = parent_position + np.array([radius, 0])
                        if i % 2 == 1:
                            nodes_pos[i] = node_pos + np.array([-0.001, -0.001])
                        else:
                            nodes_pos[i] = node_pos + np.array([0.001, 0.001])

                    elif node_location == 'west':
                        node_pos = parent_position + np.array([0, -radius])
                        if i % 2 == 1:
                            nodes_pos[i] = node_pos + np.array([-0.001, -0.001])
                        else:
                            nodes_pos[i] = node_pos + np.array([0.001, 0.001])

                    else:
                        print('location label not found')

                npos.update(zip(nodes, nodes_pos))
                radius = radius / 1.8  # reduce radius for next shell to prevent overlapping of edges

        return npos

    def draw(self,ax):
        """
        Draws the Graph on the axis provided
        :param ax: matplotlib.pyplot.axis object, axis on which the graph should be drawn
        :return: None
        """
        # Drawing the model
        pos = self.qwz_square_layout(self.shell_list())

        #Draw Nodes
        nx.draw_networkx_nodes(self.G, pos, node_size=5, ax=ax)

        # Select undirected edges
        edge_undirected = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] == self.scale]
        edge_undirected2 = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] == -self.scale]

        # Select directed edges (one direction only)
        edge_directed = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] == self.scale*1j]
        edge_directed2 = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] == -self.scale*1j]

        # Turn off arrows because they are technically undirected
        nx.draw_networkx_edges(self.G, pos, edgelist=edge_undirected, edge_color='blue', width=0.5, alpha=0.5, arrows=False,
                               ax=ax)
        nx.draw_networkx_edges(self.G, pos, edgelist=edge_undirected2, edge_color='green', alpha=0.5, width=0.5, arrows=False,
                               ax=ax)
        # Directed Edges
        nx.draw_networkx_edges(
            self.G, pos, edgelist=edge_directed, width=0.5, alpha=0.5, edge_color="orange", style="dashed", ax=ax)
        nx.draw_networkx_edges(
            self.G, pos, edgelist=edge_directed2, width=0.5, alpha=0.5, edge_color="red", style="dashed", ax=ax)




if __name__ == "__main__":
    H = HaldaneCayleyTree(4,1)
    Q = QWZCayleyTree(3)

    fig, ax_list = plt.subplots(2,1)
    fig.figsize = (15,10)

    eval, evec = H.exact_diagonalization()
    ax_list[0].hist(eval,bins=200)
    ax_list[0].set_ylabel('D')
    ax_list[0].set_xlabel('E/t')
    ax_list[0].set_title('Exact Diagonalization Spectrum with M = ' + str(H.M) + ' and N = ' + str(H.N))

    # Drawing the model
    Q.draw(ax_list[1])
    plt.tight_layout()
    plt.show()