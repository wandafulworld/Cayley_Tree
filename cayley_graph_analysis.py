import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt

# Import and immediately overwrite the nx package function
import helper_functions as tc
nx.generators.classic._tree_edges = tc._cayley_tree_edges

# ToDo: Add ax as argument for all solvers and plot internally on the ax if not None

class cayley_tree_simulation():
    def __init__(self,r,M):
        """

        :param r: Number of children per node (k = r + 1)
        :param M: Number of shells of the Cayley Tree
        """
        self._r = r
        self._M = M
        self._N = 1 + (r+1)*(2**M -1) # Number of nodes on the cayley tree

        self.G = tc.cayley_tree(r,M)
        self.A = nx.adjacency_matrix(self.G) # Sparse Matrix of the Adjacency Matrix of the Cayley Tree

    def exact_diagonalization(self, eigvals_only=False):
        """

        :param eigvals_only: Choose True if you only want the eigenvalues of the graph
        :return: w: array of N eigenvalues
        :return: v: array representing the N eigenvectors
        """
        # Note: We turn A into a dense matrix here -> limiting factor
        return sp.linalg.eigh(self.A.todense(),eigvals_only=eigvals_only) #Assumes symmetric matrix -> Might not be the case in the future

    def polynomial_solver(self):
        domain = [-2*np.sqrt(self._r),2*np.sqrt(self._r)]
        poly = [np.polynomial.polynomial.Polynomial([1],domain=domain,window=domain),np.polynomial.polynomial.Polynomial([0,1],domain=domain,window=domain)]
        x = np.polynomial.polynomial.Polynomial([0,1],domain=domain,window=domain) # For multiplication later on

        eigenval = poly[1].roots().tolist()  # List where we store the eigenvalues

        for l in range(2,self._M): # ToDo: If M < 3 this will not be triggered -> need case response for M = 1
            poly.append(x*poly[l-1] - (self._r)*poly[l-2])
            eigenval.extend(poly[l].roots().tolist())

        # Last polynomial P_L that isn't produced by the above
        poly.append(x*poly[self._M-1] - (self._r)*poly[self._M-2])
        eigenval.extend(poly[self._M].roots().tolist())
        asymmetric_polynomial = x*poly[self._M] - (self._r + 1)*poly[self._M - 1]
        eigenval.extend(asymmetric_polynomial.roots().tolist())

        return eigenval

    def _peak_finder(self,polynomial,n):
        x, y = polynomial.linspace(n)
        peaks, _ =sp.signal.find_peaks(np.abs(-y)) # Invert b/c we want minimas
        return x[peaks].tolist()

    def polynomial_minima_searcher(self,n): #ToDO: Fix to find correct eigenvalues
        domain = [-2*np.sqrt(self._r),2*np.sqrt(self._r)]
        poly = [np.polynomial.polynomial.Polynomial([1],domain=domain,window=domain),np.polynomial.polynomial.Polynomial([0,1],domain=domain,window=domain)]
        x = np.polynomial.polynomial.Polynomial([0,1],domain=domain,window=domain) # For multiplication later on

        eigenvals = []
        eigenvals.extend(self._peak_finder(poly[1],n))


        for l in range(2,self._M): # ToDo: If M < 3 this will not be triggered -> need case response for M = 1
            poly.append(x*poly[l-1] - (self._r)*poly[l-2])
            eigenvals.extend(self._peak_finder(poly[l], n))

        # Last polynomial P_L that isn't produced by the above
        poly.append(x*poly[self._M-1] - (self._r)*poly[self._M-2])
        eigenvals.extend(self._peak_finder(poly[self._M], n))
        asymmetric_polynomial = x * poly[self._M] - (self._r + 1)*poly[self._M - 1]
        eigenvals.extend(self._peak_finder(asymmetric_polynomial, n))

        return eigenvals

    def draw_cayley_tree(self,ax,color_shells=False):
        """

        :param ax: The axis on which you want to plot your cayley tree
        :param color_shells: If True, instead of the branches, the shells will have the same color
        :return: None
        """
        nlist = tc.shell_list(self.G,self._r,self._M)
        print(nlist)
        blist = tc.branch_list(self.G,self._r,self._M)
        print(blist)
        clist = tc.color_list(self._r,self._M,blist)

        if color_shells:
            clist = tc.color_list(self._r, self._M, nlist)

        nx.draw(self.G,pos=nx.shell_layout(self.G,nlist=nlist,rotate=0),ax=ax,node_shape='.',node_color=clist,cmap='tab20')









if __name__ == "__main__":
    C = cayley_tree_simulation(2,5)

    fig, ax_list = plt.subplots(3,1,sharex=True)
    fig.figsize = (15,10)

    # Exact Diagonalization
    eval, evec = C.exact_diagonalization()
    ax_list[0].hist(eval,bins=600)
    ax_list[0].set_ylabel('D')
    ax_list[0].set_title('Exact Diagonalization Spectrum')

    # Polynomial Approach
    eval_poly = C.polynomial_solver()
    ax_list[1].hist(eval_poly,bins=600)
    ax_list[1].set_ylabel('D')
    ax_list[1].set_title('Polynomial Construction Spectrum')
    ax_list[1].set_xlabel('E/t')

    # Draw Cayley Tree (Because we can, won't look good)
    C.draw_cayley_tree(ax_list[2])
    ax_list[2].set_title('Cayley Tree')

    plt.tight_layout()
    plt.show()

