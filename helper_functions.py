import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def _cayley_tree_edges(n, r):
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
                yield source, target
            except StopIteration:
                break
        if first_time:
            r = r - 1
        first_time = False

# overwrite function to make package work for cayley trees
nx.generators.classic._tree_edges = _cayley_tree_edges

def cayley_tree(r,M,create_using=None):
    """Returns the symmetric Cayley Tree.

    Parameters
    ----------
    r : int
        Branching factor of the tree; each node will have `r`
        children except for the node |0>, which will have 3.

    M : int
        Number of shells of the tree.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : NetworkX graph
        A cayley tree with coordination number r and M shells.
    """
    n = 1 + (r+1)*(2**M -1) # Number of Nodes in a Cayley Tree
    return nx.full_rary_tree(r, n, create_using=create_using)

def shell_list(G,r,M):
    """Returns a list of lists nodes, with each list containing the nodes of the same shell
        Makes use of the fact that the construction algorithm constructs each shell after the other

        Parameters
        ----------
        G : Graph Object
            The Cayley Tree of which you want the shell lists

        r : int
            Branching factor of the tree;

        M : int
            Number of shells of the tree.

        create_using : NetworkX graph constructor, optional (default=nx.Graph)
           Graph type to create. If graph instance, then cleared before populated.

        Returns
        -------
        shell_lists : Multiple lists corresponding to the number of Shells M
            each list containing the nodes of that shell
        """
    nodes = list(G.nodes)
    shell_lists = [[0]] # already contains the 0th node
    l = 1 # position of last added node
    for shell_number in range(1,M+1):
        n = (r+1)*(r**(shell_number - 1))
        shell_lists.append(nodes[l:(l+n)])
        l += n

    return shell_lists

def branch_list(G,r,M):
    """Returns a list of lists of nodes, with each list containing the nodes of the same branch
        Makes use of the fact that the construction algorithm constructs each shell after the other

        Parameters
        ----------
        G : Graph Object
            The Cayley Tree of which you want the shell lists

        r : int
            Branching factor of the tree;

        M : int
            Number of shells of the tree.

        create_using : NetworkX graph constructor, optional (default=nx.Graph)
           Graph type to create. If graph instance, then cleared before populated.

        Returns
        -------
        branch_list : Multiple lists corresponding to the number of branches r+1
            each list containing the nodes of that branch
        """
    nodes = list(G.nodes)
    branch_lists = [[] for _ in range(r+1)] #create lists for each branch
    l = 1 # position of last added node -> 0th node is not part of any branch
    for shell_number in range(1,M+1):
        n = r**(shell_number - 1) # Number of nodes per shell per branch
        for branch_number in range(r+1):
            branch_lists[branch_number].append(nodes[l:(l+n)])
            l += n

    for branch_number in range(r+1):
        # flatten the lists for convenience
        a = branch_lists[branch_number]
        branch_lists[branch_number] = [item for row in a for item in row]
    return branch_lists


def color_list(r,M,nlists):
    """Returns a list of color gradients (0,1) where the list positions correspond to node position given through the nlists.
        Each node list in nlists will get a different color gradient that can then be mapped using a cmap

        Parameters
        ----------
        r : int
            Branching factor of the tree;

        M : int
            Number of shells of the tree.

        nlists : A list of lists of nodes of a Graph. Each sublist will get a color assigned.

        Returns
        -------
        color_list : One list of color positions (between 0 and 1), where all the nodes of the same sublist will have the same color position
            This lists then needs to be mapped to a colormap via the draw function of networkx
            The 0th node will always be set to 0
        """
    # Determine number of colors needed
    n_colors = len(nlists)
    color_pos = np.linspace(0,1,n_colors+1)

    N = 1 + (r+1)*(2**M -1) # Number of Nodes in a Cayley Tree
    color_array = np.zeros(N)
    for i,sublist in enumerate(nlists):
        np.put(color_array,ind=sublist,v=color_pos[i+1])

    return color_array.tolist()

if __name__ == "__main__":
    r = 2
    M = 4

    G = cayley_tree(r,M)
    print(list(G.nodes))
    # Use the shell list to position nodes on shell in the drawing program
    nlist = shell_list(G,r,M)
    print(nlist)
    blist = branch_list(G,r,M)
    print(blist)
    clist = color_list(r,M,blist)

    nx.draw(G,pos=nx.shell_layout(G,nlist=nlist,rotate=0),node_shape='.',node_color=clist,cmap='tab20')
    plt.show()

# ToDo rewrite shell_layout function such that it positions the branches a bit more symmetrically