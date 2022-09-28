import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(TN):

    """Draws network x graph for tensor network defiend as a tuple of lists
    -first element is teh list of tensors
    -seconf element is teh list of connects"""

    tensors = TN[0]
    connects = TN[1]

    G = nx.Graph()
    G.add_nodes_from(range(len(tensors)))
    G.add_edges_from([(1,2)])

    nx.draw(G)
    plt.show()
