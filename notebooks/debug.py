import bhtsne
import numpy as np
import sklearn.manifold
import scipy
from sklearn.metrics.pairwise import pairwise_distances

# Quadtree checks:
# X Num of particles conserved
# X CoM of root nodes = CoM of particles
# X Num of leaves with particles when freed = Num of particles
# X Num of cells freed = num of cells created
# X t-SNE forces make sense for just a few points, like N=5
#   t-SNE forces are approximately the same when theta=0.5 and theta=0
#   t-SNE forces are approx the same between parallel case and single
#   Changes tests to sklearn style
# This doesn't involve van der Maaten's 2nd perlexity method, which only
# calculates pij over nearest neighbors


def test1():
    """ This case with two particles test the tree with only a single
        set of children
    """
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0]])
    pos_output = np.array([[-4.961291e-05, -1.072243e-04],
                           [9.259460e-05, 2.702024e-04]])
    grad_output = np.array([[-2.37012478e-05, -6.29044398e-05],
                            [2.37012478e-05, 6.29044398e-05]])
    test(pos_input, pos_output, grad_output)


def test2():
    """ This case with two particles test the tree with
        multiple levels of children
    """
    pos_input = np.array([[1.0, 0.0], [0.0, 1.0],
                          [5.0, 2.0], [7.3, 2.2]])
    pos_output = np.array([[6.080564e-05, -7.120823e-05],
                           [-1.718945e-04, -4.000536e-05],
                           [-2.271720e-04, 8.663310e-05],
                           [-1.032577e-04, -3.582033e-05]])
    grad_output = np.array([[5.81128448e-05, -7.78033454e-06],
                            [-5.81526851e-05, 7.80976444e-06],
                            [4.24275173e-08, -3.69569698e-08],
                            [-2.58720939e-09, 7.52706374e-09]])
    test(pos_input, pos_output, grad_output)


def test3():
    """ This tests enough data to be using summary nodes
    """
    pos_input = np.loadtxt(open("t64.csv", "r"))
    pos_output = np.loadtxt(open("t64.rand.csv", "r"), delimiter=',')
    grad_output = np.loadtxt(open("t64.grad.csv", "r"), delimiter=',')

    test(pos_input, pos_output, grad_output, perplexity=10.0)


def test_parallel():
    pass


def tree_consistency(verbose=False):
    """ Ensure tree-level sanity. Checks that the number of particles
        entered and the number freed are equal.
    """
    pos_output = np.loadtxt(open("t64.rand.csv", "r"), delimiter=',')
    consistency_check = bhtsne.consistency_checks(pos_output, verbose=verbose)
    assert consistency_check


def test(pos_input, pos_output, grad_output, verbose=False, perplexity=0.1):
    distances = pairwise_distances(pos_input)
    args = distances, perplexity, verbose
    pij_input = sklearn.manifold.t_sne._joint_probabilities(*args)
    pij_input = scipy.spatial.distance.squareform(pij_input)

    grad_bh, grad_exact = bhtsne.quadtree_compute(pij_input, pos_output,
                                                  verbose=verbose)
    assert np.allclose(grad_bh, grad_output, 1e-4), (grad_bh, grad_output)
    return grad_bh

if __name__ == '__main__':
    test3()
    test1()
    test2()
    tree_consistency(verbose=False)
