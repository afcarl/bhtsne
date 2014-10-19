import numpy as np
cimport numpy as np
cimport cython

from stdlib cimport malloc, free, abs
from fp_utils cimport imax, fmax, imin, fmin, iclip, fclip

# Need a recursive Barnes-Hut walker
# yields every node below the theta size threshold

# Need a method that computes the running distance normalization
# taking in the BH-nodes that the walker yields

cdef struct QuadNode:
	# Instead of keeping track of the center of mass
	# we keep the cumulative sum of all particle positions
	# To get the CoM later we divide by the number of particles
	np.float64_t cum_pos[2]
	# If this is a leaf, the position of the particle within this leaf 
	np.float64_t cur_pos[2]
	# The number of particles including all 
	# nodes below this one
	np.int64_t np
	# level = 0 is the root node
	# And each subdivision adds 1 to the level
	np.int32_t level = 0
	# Left edge of this node, normalized to [0,1]
	np.float64_t le[2] = [0., 0.]
	# The width of this node -- used to calculate the opening
	# angle. Equal to width = re - le
	np.float64_t w = 0.

	# Does this node have children?
	# Default to leaf until we add particles
	np.uint8_t is_leaf = True
	# Keep pointers to the child nodes
	QuadNode *children[2][2]
	# Keep a pointer to the parent
	QuadNode *parent
	# Keep a pointer to the node
	# to visit next when visting every QuadNode
	# QuadNode *next

# Need a method for recursive insertion
# creates QNodes, and keeps track of the CoM
# of each node

cdef QuadNode create_root():
	cdef int ax
	cdef np.floate64_t w = parent.width / 2.0
	root = <OctreeNode *> malloc(sizeof(OctreeNode))
	root.is_leaf = True
	root.parent = None
	root.level = 0
	for ax in range(2):
		root.le[ax] = 0.
		root.re[ax] = 1.
		root.c[ax]  = 0.5
		root.cum_pos[ax] = 0.
		root.com[ax] = 0.
	root.width = 1.
	root.np = 0
	root.np_local = 0
	return root

cdef inline *QuadNode select_child(QuadNode *node, np.float64_t pos[2]):
	# Find which sub-node a position should go into
	# And return the appropriate node
	cdef np.floate64_t offset[2] = (pos - node.c) > 0.
	return node.children[offset[0]][offset[1]]

cdef QuadNode create_child(QuadNode *parent, np.int64_t offset):
	cdef int ax
	cdef np.floate64_t w = parent.width / 2.0
	child = <OctreeNode *> malloc(sizeof(OctreeNode))
	child.is_leaf = True
	child.parent = parent
	child.level = parent.level + 1
	for ax in range(2):
		child.le[ax] = parent.le[ax] + offset[ax] * w
		child.re[ax] = child.le[ax] + w
		child.c[ax] = (child.le[ax] + child.re[ax]) /2.0
		child.cum_pos[ax] = 0.
		child.com[ax] = 0.
	child.width = w 
	child.np = 0
	child.np_local = 0
	return child

cdef subdivide(QuadNode *node):
	# This instantiates 4 nodes for the current node
	node.is_leaf = False
	cdef int i = 0
	cdef int j = 0
	cdef np.float64_t w = node.width / 2.0
	cdef np.int64_t offset[2]
	for i in range(2):
		offset[i][j] = i
		for j in range(2):
			offset[i][j] = j
			node.children[i][j] = create_child(node, offset)

cdef insert(QuadNode *root, np.float64_t pos[2])
	cdef QuadNode *child
	cdef np.float64_t p[2]
	root.cum_pos += pos
	root.np += 1
	if np.isnan(root.cur_pos[0]) & root.is_leaf:
		root.cu
	else:
		if root.is_leaf:
			subdivide(root)
		for p in [root.cur_pos, pos]:
			child = select_child(root, p)	
			insert(child, p)
		for i in range(2): 
			root.cur_pos[i] = np.nan
		