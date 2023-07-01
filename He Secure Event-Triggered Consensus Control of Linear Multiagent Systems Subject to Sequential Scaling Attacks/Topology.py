import numpy as np
import networkx as nx
import scipy

import Constants

# G

G = nx.Graph()
informed_nodes = (1, 2, 3, 5, 8, 9, 12, 13, 15, 17, 18)
G_adj_mat = np.array([
    (0, 1), (0, 2), (0, 3), (0, 5), (0, 8), (0, 9), (0, 12), (0, 13), (0, 15), (0, 17), (0, 18),
    (1, 9), (1, 13), (1, 14), (1, 19),
    (2, 14), (2, 15), (2, 16),
    (3, 4), (3, 7), (3, 16), (3, 20),
    (4, 6), (4, 8), (4, 15),
    (5, 6), (5, 8), (5, 9), (5, 10), (5, 11), (5, 13), (5, 18),
    (6, 10), (6, 11),
    (7, 10), (7, 13), (7, 16),
    (8, 9), (8, 10), (8, 14), (8, 15), (8, 16), (8, 20),
    (9, 14), (9, 15), (9, 16),
    (10, 17),
    (11, 18),
    (12, 15), (12, 19), (12, 20),
    (13, 15), (13, 17), (13, 18), (13, 20),
    (14, 19), (14, 20),
    (15, 19),
    (16, 20),
    (17, 18), (17, 19),
    (18, 19),
])
G.add_nodes_from(range(Constants.N + 1))
G.add_edges_from(G_adj_mat)

# Sub G
sub_G = nx.subgraph(G, range(1, Constants.N + 1))
sub_G_adj_mat = nx.to_numpy_array(sub_G)

# Mat
L_adj_mat = scipy.sparse.csgraph.laplacian(sub_G_adj_mat)

informed_symbols = []
for i in range(Constants.N):
    if np.isin(i, informed_nodes):
        informed_symbols.append(1)
    else:
        informed_symbols.append(0)

D_mat = np.diag(informed_symbols)

H_adj_mat = L_adj_mat + D_mat

eig = np.linalg.eig(H_adj_mat)