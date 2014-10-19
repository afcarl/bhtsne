# K = BH theta angle


# Normalize input s.t. max(x) = 1

# Compute perplexity for every point
	# Build distance tree
	# For each datapoint
	# Compute gaussian kernel row over K nearest neighbors
	# This is equivalent to calculating the likelihood
	# of seeing each datapoint given the width of the gaussian
	# around the central point
	# Then we compute teh entropy of these likelihoods
	# gaussian 
		# Hsum = sum beta * d[i] * exp( -beta * d[i]) 
		# P[i] = 	  		    	exp( -beta * d[i])
		# H = Hsum / P.sum() + log(P.sum())
		# H   = sum[ebd[j] * bdj[j]] / sum_p + log[sum_p]
		# if entropy is higher/lower than the desired perplexity
		# adjust beta up or down accordingly
	# save nearest neighbor indices, as well as normed
	# p_ij

# Symmetrize matrix

# Initialize random

# Compute Gradient 

# Apply gradient

# Constrain to zero mean

# bail out when converged


# NOTES

# 
		# H[i] = sum [p log p ]
		# p[i] = exp(-beta * d) / sum_p
		# H[i] = sum[ebd[j] / sum_p * log ebd[j] / sump]
		# H[i] = sum[ebd[j] / sum_p * (log ebd[j] - log sump])
		# H[i] = sum[ebd[j] (-bd[j] / sum_p - log [sum_p] / sump)
		# H[i] = sum[ebd[j] -bd[j] / sump_p] - N * log[sum_p] / sum_p
		# H[i] = sum[ebd[j] * -bd[j] / sum_p] - N log[sum_p] / sum_p
