import bhtsne
import numpy as np
import sklearn.manifold
import scipy
from sklearn.metrics.pairwise import pairwise_distances

pos_input  = np.array([[1.0, 0.0],[0.0, 1.0]])
pos_output = np.array([[-4.961291e-05, -1.072243e-04],[9.259460e-05, 2.702024e-04]])

pos_rand = np.random.random((10,2))
pos_rand -= pos_rand.mean(axis=0)

distances = pairwise_distances(pos_input)
pij_input = sklearn.manifold.t_sne._joint_probabilities(distances, 0.1, True)
pij_input = scipy.spatial.distance.squareform(pij_input)

grad = bhtsne.quadtree_compute(pij_input, pos_output, verbose=1)
