import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
cimport numpy as np
cimport cython

# Implementation by Chris Moody
# original code by Laurens van der Maaten
# for reference implemenations and papers describing the technique:
# http://homepage.tudelft.nl/19j49/t-SNE.html

# Quadtree checks:
# X Num of particles conserved
# X CoM of root nodes = CoM of particles
# X Num of leaves with particles when freed = Num of particles
# X Num of cells freed = num of cells created
#   t-SNE forces make sense for just a few points, like N=5
#   t-SNE forces are approximately the same when theta=0.5 and theta=0
#   t-SNE forces are approximately the same between BH case and the N^3 calculation

# TODO:
# Include usage documentation
# Find all references to embedding in sklearn & see where else we can document
# Incorporate into SKLearn
# PEP8 the code
# Am I using memviews or returning fulla rrays appropriately?
# DONE:
# Cython deallocate memory

from libc.stdlib cimport malloc, free, abs

cdef extern from "math.h":
    double sqrt(double x) nogil

cdef struct QuadNode:
    # Keep track of the center of mass
    np.float64_t cum_com[2]
    # If this is a leaf, the position of the particle within this leaf 
    np.float64_t cur_pos[2]
    # The number of particles including all 
    # nodes below this one
    np.uint64_t cum_size
    # Number of particles at this node
    np.uint64_t size
    # Index of the particle at this node
    np.uint64_t point_index
    # level = 0 is the root node
    # And each subdivision adds 1 to the level
    np.uint64_t level
    # Left edge of this node, normalized to [0,1]
    np.float64_t le[2] 
    # The center of this node, equal to le + w/2.0
    np.float64_t c[2] 
    # The width of this node -- used to calculate the opening
    # angle. Equal to width = re - le
    np.float64_t w[2]

    # Does this node have children?
    # Default to leaf until we add particles
    np.uint8_t is_leaf
    # Keep pointers to the child nodes
    QuadNode *children[2][2]
    # Keep a pointer to the parent
    QuadNode *parent
    # Keep a pointer to the node
    # to visit next when visting every QuadNode
    # QuadNode *next

cdef class QuadTree:
    # Holds a pointer to the root node
    cdef QuadNode* root_node 
    # Total number of cells
    cdef int num_cells
    # Total number of particles
    cdef int num_part
    # Spit out diagnostic information?
    cdef int verbose
    
    def __cinit__(self, verbose=False, width=None):
        if width is None:
            width = np.array([1., 1.])
        print("creating root")
        self.root_node = self.create_root(width)
        print("done")
        self.num_cells = 0
        self.num_part = 0
        self.verbose = verbose

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline QuadNode* create_root(self, np.ndarray[np.float64_t, ndim=1] width):
        # Create a default root node
        cdef int ax
        root = <QuadNode*> malloc(sizeof(QuadNode))
        root.is_leaf = True
        root.parent = NULL
        root.level = 0
        root.cum_size = 0
        root.size = 0
        root.point_index = -1
        for ax in range(2):
            root.w[ax] = width[ax]
            root.le[ax] = 0. - root.w[ax] / 2.0
            root.c[ax] = 0.0
            root.cum_com[ax] = 0.
            root.cur_pos[ax] = -1.
        self.num_cells += 1
        return root

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline QuadNode* create_child(self, QuadNode *parent, np.uint8_t offset[2]):
        # Create a new child node with default parameters
        cdef int ax
        child = <QuadNode *> malloc(sizeof(QuadNode))
        child.is_leaf = True
        child.parent = parent
        child.level = parent.level + 1
        child.size = 0
        child.cum_size = 0
        child.point_index = -1
        for ax in range(2):
            child.w[ax] = parent.w[ax] / 2.0
            child.le[ax] = parent.le[ax] + offset[ax] * parent.w[ax] / 2.0
            child.c[ax] = child.le[ax] + child.w[ax] / 2.0
            child.cum_com[ax] = 0.
            child.cur_pos[ax] = -1.
        self.num_cells += 1
        return child

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline QuadNode* select_child(self, QuadNode *node, np.float64_t pos[2]):
        # Find which sub-node a position should go into
        # And return the appropriate node
        cdef int offset[2]
        cdef int ax
        for ax in range(2):
            offset[ax] = (pos[ax] - (node.le[ax] + node.w[ax] / 2.0)) > 0.
        return node.children[offset[0]][offset[1]]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void subdivide(self, QuadNode *node):
        # This instantiates 4 nodes for the current node
        cdef int i = 0
        cdef int j = 0
        cdef np.uint8_t offset[2] 
        node.is_leaf = False
        offset[0] = 0
        offset[1] = 0
        for i in range(2):
            offset[0] = i
            for j in range(2):
                offset[1] = j
                node.children[i][j] = self.create_child(node, offset)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void insert(self, QuadNode *root, np.float64_t pos[2], int point_index):
        # Introduce a new point into the quadtree
        # by recursively inserting it and subdividng as necessary
        cdef QuadNode *child
        cdef int i
        cdef int ax
        # Increment the total number points including this
        # node and below it
        root.cum_size += 1
        # Evaluate the new center of mass, weighting the previous
        # center of mass against the new point data
        cdef double frac_seen = <double>(root.cum_size - 1) / (<double> root.cum_size)
        cdef double frac_new  = 1.0 / <double> root.cum_size
        for ax in range(2):
            root.cum_com[ax] *= frac_seen
        for ax in range(2):
            root.cum_com[ax] += pos[ax] * frac_new
        # If this node is unoccupied, fill it.
        # Otherwise, we need to insert recursively.
        if self.verbose:
            print('%i' % point_index, 'level', root.level, 'cum size is', root.cum_size)
        #if root.level > 4:
        #    print 'MAX LEVEL EXCEEDED'
        #    return
        # Two insertion scenarios: 
        # 1) Insert into this node if it is a leaf and empty
        # 2) Subdivide this node if it is currently occupied
        if (root.size == 0) & root.is_leaf:
            # print('%i' % point_index, 'inserting into leaf')
            for ax in range(2):
                root.cur_pos[ax] = pos[ax]
            root.point_index = point_index
            root.size = 1
        else:
            # If necessary, subdivide this node before
            # descending
            if root.is_leaf:
                if self.verbose:
                    print('%i' % point_index, 'subdividing', 
                          root.cum_com[0], root.cum_com[1])
                self.subdivide(root)
            # We have two points to relocate: the one previously
            # at this node, and the new one we're attempting
            # to insert
            if root.size > 0:
                # print('%i' % root.point_index, 'selecting child for previous')
                child = self.select_child(root, root.cur_pos)
                # print('%i' % root.point_index, 'inserting for previous')
                self.insert(child, root.cur_pos, root.point_index)
                # Remove the point from this node
                for ax in range(2):
                    root.cur_pos[ax] = -1
                root.size = 0
                root.point_index = -1
            # Insert the new point
            # print('%i' % point_index, 'selecting for new')
            child = self.select_child(root, pos)
            # print('%i' % point_index, 'inserting for new')
            self.insert(child, pos, point_index)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void insert_many(self, np.ndarray[np.float64_t, ndim=2] pos_array):
        cdef int nrows = pos_array.shape[0]
        cdef int i, ax
        cdef np.float64_t row[2]
        for i in range(nrows):
            for ax in range(2):
                row[ax] = pos_array[i, ax]
            # print("inserting point %i" % i)
            self.insert(self.root_node, row, i)
            self.num_part += 1
        if self.verbose:
            print("tree COM %1.3e %1.3e" % (self.root_node.cum_com[0], self.root_node.cum_com[1]))
            print("part COM %1.3e %1.3e" % (pos_array[:,0].mean(), pos_array[:,1].mean()))

    def free(self):
        cdef np.ndarray cnt = np.zeros(3, dtype='int32')
        self.free_recursive(self.root_node, cnt)
        free(self.root_node)
        if self.verbose:
            print("   freed %i cells out of %i" % (cnt[0], self.num_cells))
            print("   freed %i leaves with particles of %i" % (cnt[2], self.num_part))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void free_recursive(self, QuadNode *root, np.ndarray[np.int32_t, ndim=1] counts):
        cdef int i, j    
        cdef QuadNode* child
        if not root.is_leaf:
            for i in range(2):
                for j in range(2):
                    child = root.children[i][j]
                    self.free_recursive(child, counts)
                    counts[0] += 1
                    if child.is_leaf:
                        counts[1] += 1
                        if child.size > 0:
                            counts[2] +=1
                    free(child)

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #@cython.cdivision(True)
    cdef np.ndarray compute_gradient(self, np.float64_t theta,
                                     np.ndarray[np.float64_t, ndim=2] val_P,
                                     np.ndarray[np.float64_t, ndim=2] pos_reference):
        cdef int n = pos_reference.shape[0]
        cdef np.ndarray pos_force = np.zeros((n, 2), dtype='float64')
        cdef np.ndarray neg_force = np.zeros((n, 2), dtype='float64')
        cdef np.uint64_t point_index
        self.compute_edge_forces(val_P, pos_reference, pos_force)
        for point_index in range(pos_reference.shape[0]):
            sum_Q = self.compute_non_edge_forces(self.root_node, theta, sum_Q, point_index,
                                                pos_reference, neg_force)
            if self.verbose:
                print "neg_force[%i,:]=" % point_index, neg_force[point_index, :]
        return pos_force - (neg_force / sum_Q)

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #@cython.cdivision(True)
    cdef np.ndarray compute_gradient_exact(self, np.float64_t theta,
                                           np.ndarray[np.float64_t, ndim=2] val_P,
                                           np.ndarray[np.float64_t, ndim=2] pos_reference):
        cdef np.float64_t sum_Q = 0.0
        cdef np.float64_t mult = 0.0
        cdef int i, j

        cdef int n = pos_reference.shape[0]
        cdef np.ndarray dC = np.zeros((n, 2), dtype='float64')
        cdef np.ndarray DD = pairwise_distances(pos_reference)
        cdef np.ndarray  Q = np.zeros((n, n), dtype='float64')

        # Computation of the Q matrix & normalization sum
        for i in range(pos_reference.shape[0]):
            for j in range(pos_reference.shape[0]):
                Q[i, j] = 1. / (1. + DD[i, j])
                sum_Q += Q[i, j]

        # Computation of the gradient
        for i in range(pos_reference.shape[0]):
            for j in range(pos_reference.shape[0]):
                if i == j: 
                    continue
                mult = (val_P[i, j] - (Q[i, j] / sum_Q)) * Q[i, j]
                for ax in range(2):
                    dC[i, ax] += mult * (pos_reference[i, ax] - pos_reference[j, ax])
        return dC
        

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #@cython.cdivision(True)
    cdef void compute_edge_forces(self, np.ndarray[np.float64_t, ndim=2] val_P,
                             np.ndarray[np.float64_t, ndim=2] pos_reference,
                             np.ndarray[np.float64_t, ndim=2] force):
        # Sum over the following expression for i not equal to j
        # p_ij Q (1 + ||y_i - y_j||^2)^-1 (y_i - y_j)
        cdef int i, j, dim
        cdef np.float64_t buff[2]
        cdef np.float64_t D
        for i in range(pos_reference.shape[0]):
            for j in range(pos_reference.shape[1]):
                if i == j : 
                    continue
                D = 0.0
                for dim in range(2):
                    buff[dim] = pos_reference[i, dim] - pos_reference[j, dim]
                    D += buff[dim] ** 2.0  
                D = val_P[i, j] / (1.0 + D)
                for dim in range(2):
                    force[i, dim] += D * buff[dim]

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #@cython.cdivision(True)
    cdef double compute_non_edge_forces(self, QuadNode* node, 
                                 np.float64_t theta,
                                 double sum_Q,
                                 np.uint64_t point_index,
                                 np.ndarray[np.float64_t, ndim=2] pos_reference,
                                 np.ndarray[np.float64_t, ndim=2] force):
        # Compute the t-SNE force on the point in pos_reference given by point_index
        cdef QuadNode* child
        cdef int i, j
        cdef np.float64_t dist2, mult, qijZ
        cdef np.float64_t delta[2] 
        cdef np.float64_t wmax = 0.0

        for i in range(2):
            delta[i] = 0.0

        # There are no points below this node if cum_size == 0
        # so do not bother to calculate any force contributions
        # Also do not compute self-interactions
        if node.cum_size > 0 and not (node.is_leaf and (node.point_index == point_index)):
            dist2 = 0.0
            # Compute distance between node center of mass and the reference point
            for i in range(2):
                delta[i] += pos_reference[point_index, i] - node.cum_com[i] 
                if self.verbose:
                    print "pos_reference[point_index=%i,i=%i]=%1.6e" % (point_index, i, pos_reference[point_index,i])
                    print "node.cum_com[i=%i]=%1.1e" % (i, node.cum_com[i])
                    print "delta[i=%i]=%1.6e" % (i, delta[i]) 
                dist2 += delta[i]**2.0
            if self.verbose:
                print "dist2=%1.6e" % dist2
            # Check whether we can use this node as a summary
            # It's a summary node if the angular size as measured from the point
            # is relatively small (w.r.t. to theta) or if it is a leaf node.
            # If it can be summarized, we use the cell center of mass 
            # Otherwise, we go a higher level of resolution and into the leaves.
            for i in range(2):
                wmax = max(wmax, node.w[i])

            if node.is_leaf or (wmax / sqrt(dist2) < theta):
                # Compute the t-SNE force between the reference point and the current node
                qijZ = 1.0 / (1.0 + dist2)
                sum_Q += node.cum_size * qijZ
                mult = node.cum_size * qijZ * qijZ
                for ax in range(2):
                    force[point_index, ax] += mult * delta[ax]
            else:
                # Recursively apply Barnes-Hut to child nodes
                for i in range(2):
                    for j in range(2):
                        child = node.children[i][j]
                        sum_Q += self.compute_non_edge_forces(child, theta, sum_Q, 
                                                              point_index,
                                                              pos_reference, force)
            return sum_Q
        
    cdef void check_consistency(self):
        cdef int count 
        count = 0
        count = self.count_points(self.root_node, count)
        if self.verbose:
            print(" counted %i points" % count)
            print("    root %i points" % self.root_node.cum_size)
            print("    tree %i points" % self.num_part)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int count_points(self, QuadNode* root, int count):
        # Walk through the whole tree and count the number 
        # of points at the leaf nodes
        cdef QuadNode* child
        cdef int i, j
        for i in range(2):
            for j in range(2):
                child = root.children[i][j]
                if child.is_leaf and child.size > 0:
                    count += 1
                elif not child.is_leaf:
                    count = self.count_points(child, count)
                # else case is we have an empty leaf node
                # which happens when we create a quadtree for
                # one point, and then the other neighboring cells
                # don't get filled in
        return count

cdef QuadTree create_quadtree(pos_output, verbose=0):
    # pos_output -= pos_output.mean(axis=0)
    width = pos_output.max(axis=0) - pos_output.min(axis=0)
    qt = QuadTree(verbose=verbose, width=width)
    qt.insert_many(pos_output)
    qt.check_consistency()
    return qt

def quadtree_compute(pij_input, pos_output, theta=0.5, verbose=0):
    qt = create_quadtree(pos_output, verbose=verbose)
    forces1 = qt.compute_gradient(theta, pij_input, pos_output)
    forces2 = qt.compute_gradient_exact(theta, pij_input, pos_output)
    qt.free()
    return forces1, forces2
