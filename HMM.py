import random
import numpy as np

########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random

class HiddenMarkovModel:
	'''
	Class implementation of Hidden Markov Models.
	'''

	def __init__(self, A, O):
		'''
		Initializes an HMM. Assumes the following:
			- States and observations are integers starting from 0. 
			- There is a start state (see notes on A_start below). There
			  is no integer associated with the start state, only
			  probabilities in the vector A_start.
			- There is no end state.

		Arguments:
			A:          Transition matrix with dimensions L x L.
						The (i, j)^th element is the probability of
						transitioning from state i to state j. Note that
						this does not include the starting probabilities.

			O:          Observation matrix with dimensions L x D.
						The (i, j)^th element is the probability of
						emitting observation j given state i.

		Parameters:
			L:          Number of states.
			
			D:          Number of observations.
			
			A:          The transition matrix.
			
			O:          The observation matrix.
			
			A_start:    Starting transition probabilities. The i^th element
						is the probability of transitioning from the start
						state to state i. For simplicity, we assume that
						this distribution is uniform.
		'''

		self.L = len(A)
		self.D = len(O[0])
		self.A = A
		self.O = O
		self.A_start = [1. / self.L for _ in range(self.L)]


	def viterbi(self, x):
		'''
		Uses the Viterbi algorithm to find the max probability state 
		sequence corresponding to a given input sequence.

		Arguments:
			x:          Input sequence in the form of a list of length M,
						consisting of integers ranging from 0 to D - 1.

		Returns:
			max_seq:    State sequence corresponding to x with the highest
						probability.
		'''

		M = len(x)      # Length of sequence.

		# The (i, j)^th elements of probs and seqs are the max probability
		# of the prefix of length i ending in state j and the prefix
		# that gives this probability, respectively.
		#
		# For instance, probs[1][0] is the probability of the prefix of
		# length 1 ending in state 0.
		probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
		seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

		# Initializing base case: length 1 states (row 1)
		for s in range(self.L):
			probs[1][s] = self.A_start[s] * self.O[s][x[0]]
			seqs[1][s] = str(s)

		# For each row from index 2 (length 2) to index M (length M):
		# For each 'length' prefix (row 'length'):
		for length in range(2, M+1):
			# For each possible end state of the prefix:
			for curr_state in range(self.L):
				max_prob = 0.0
				best_seq = ''
				# For each previous prefix of length 'length-1', find max prob:
				for prev_state in range(self.L):
					curr_prob = probs[length-1][prev_state] * self.A[prev_state][curr_state] * self.O[curr_state][x[length-1]]
					if curr_prob > max_prob:
						max_prob = curr_prob
						best_seq = seqs[length-1][prev_state] + str(curr_state)
				
				# Stores max_prob and best_seq into their respective matrices:
				probs[length][curr_state] = max_prob
				seqs[length][curr_state] = best_seq

		# Finds sequence w/ max probability from last row:
		best_index = 0
		max_prob = 0.0
		for s in range(self.L):
			if probs[M][s] > max_prob:
				best_index = s
				max_prob = probs[M][s]

		max_seq = seqs[M][best_index]
		return max_seq


	def forward(self, x, normalize=False):
		'''
		Uses the forward algorithm to calculate the alpha probability
		vectors corresponding to a given input sequence.

		Arguments:
			x:          Input sequence in the form of a list of length M,
						consisting of integers ranging from 0 to D - 1.

			normalize:  Whether to normalize each set of alpha_j(i) vectors
						at each i. This is useful to avoid underflow in
						unsupervised learning.

		Returns:
			alphas:     Vector of alphas.

						The (i, j)^th element of alphas is alpha_j(i),
						i.e. the probability of observing prefix x^1:i
						and state y^i = j.

						e.g. alphas[1][0] corresponds to the probability
						of observing x^1:1, i.e. the first observation,
						given that y^1 = 0, i.e. the first state is 0.
		'''

		M = len(x)      # Length of sequence.
		alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

		# Initiate row 1 (skip row 0) of alphas array:

		for s in range(self.L):
			# print(s)
			alphas[1][s] = self.O[s][x[0]] * self.A_start[s]

		# Fill in all other rows of alphas:
		
		# For each 'length' of prefix:
		for length in range(2, M+1):
			# For each end state:
			for curr_state in range(self.L):
				# Sum over all previous states:
				total_sum = 0.0
				for prev_state in range(self.L):
					total_sum += alphas[length-1][prev_state] * self.A[prev_state][curr_state]
				# Multiply sum by entry in observation matrix and add to alphas:
				alphas[length][curr_state] = self.O[curr_state][x[length-1]] * total_sum

		if normalize:
			for i in range(1, M+1):
				if np.sum(alphas[i]) == 0:
					continue
				alphas[i] /= np.sum(alphas[i])

		return alphas


	def backward(self, x, normalize=False):
		'''
		Uses the backward algorithm to calculate the beta probability
		vectors corresponding to a given input sequence.

		Arguments:
			x:          Input sequence in the form of a list of length M,
						consisting of integers ranging from 0 to D - 1.

			normalize:  Whether to normalize each set of alpha_j(i) vectors
						at each i. This is useful to avoid underflow in
						unsupervised learning.

		Returns:
			betas:      Vector of betas.

						The (i, j)^th element of betas is beta_j(i), i.e.
						the probability of observing suffix x^(i+1):M and
						state y^i = j.

						e.g. betas[M][0] corresponds to the probability
						of observing x^M+1:M, i.e. no observations,
						given that y^M = 0, i.e. the last state is 0.
		'''

		M = len(x)      # Length of sequence.
		betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

		# Initiate row M of betas array:
		for s in range(self.L):
			betas[M][s] = 1.0

		# Fill in all other rows of betas:
		
		# For each 'length' of prefix:
		for length in range(M-1, -1, -1):
			# For each beginning state:
			for curr_state in range(self.L):
				# Sum over all 'next' states:
				total_sum = 0.0
				for next_state in range(self.L):
					if length == 0:
						total_sum += betas[length+1][next_state] * self.A_start[next_state] * self.O[next_state][x[length]]
					total_sum += betas[length+1][next_state] * self.A[curr_state][next_state] * self.O[next_state][x[length]]

				# Add to betas:
				betas[length][curr_state] = total_sum

		if normalize:
			for i in range(M+1):
				if np.sum(betas[i]) == 0:
					continue
				betas[i] /= np.sum(betas[i])

		return betas


	def supervised_learning(self, X, Y):
		'''
		Trains the HMM using the Maximum Likelihood closed form solutions
		for the transition and observation matrices on a labeled
		datset (X, Y). Note that this method does not return anything, but
		instead updates the attributes of the HMM object.

		Arguments:
			X:          A dataset consisting of input sequences in the form
						of lists of variable length, consisting of integers 
						ranging from 0 to D - 1. In other words, a list of
						lists.

			Y:          A dataset consisting of state sequences in the form
						of lists of variable length, consisting of integers 
						ranging from 0 to L - 1. In other words, a list of
						lists.

						Note that the elements in X line up with those in Y.
		'''

		# Calculate each element of A using the M-step formulas.
		# (A is transition matrix)
		N = len(X)
		for prev_state in range(self.L):
			test = 0.0
			for curr_state in range(self.L):
				numer = 0
				# Count number of transitions from prev_state to curr_state:
				for i in range(N):
					for j in range(len(Y[i])-1): # have to subtract 1?
						if Y[i][j] == prev_state and Y[i][j+1] == curr_state:
							numer += 1
				denom = 0
				# Count number of occurrences of prev_state:
				for i in range(N):
					for j in range(len(Y[i])-1):
						if Y[i][j] == prev_state:
							denom += 1
				# Update A matrix:
				self.A[prev_state][curr_state] = float(numer) / float(denom)
				test += float(numer) / float(denom)

		# Calculate each element of O using the M-step formulas.
		# (O is observation matrix)
		for state in range(self.L):
			test = 0.0
			for obs in range(self.D):
				numer = 0
				# Count number of occurrences of obs and state:
				for i in range(N):
					for j in range(len(Y[i])):
						if X[i][j] == obs and Y[i][j] == state:
							numer += 1
				denom = 0
				# Count number of occurrences of state:
				for i in range(N):
					for j in range(len(Y[i])):
						if Y[i][j] == state:
							denom += 1
				# Update O matrix:
				self.O[state][obs] = float(numer) / float(denom)
				test += float(numer) / float(denom)


	def unsupervised_learning(self, X, N_iters):
		'''
		Trains the HMM using the Baum-Welch algorithm on an unlabeled
		datset X. Note that this method does not return anything, but
		instead updates the attributes of the HMM object.

		Arguments:
			X:			A dataset consisting of input sequences in the form
						of lists of length M, consisting of integers ranging
						from 0 to D - 1. In other words, a list of lists.

			N_iters:	The number of iterations to train on.
		'''

		# Transition and Observation matrices already initialized.
		for iters in range(N_iters):
			if (iters % 10) == 0:
				print(iters)	
			numerator_A = [[0.0 for _ in range(self.L)] for _ in range(self.L)]
			numerator_O = [[0.0 for _ in range(self.D)] for _ in range(self.L)]
			denom_A = [[0.0 for _ in range(self.L)] for _ in range(self.L)]		# ADDED
			denom_O = [[0.0 for _ in range(self.D)] for _ in range(self.L)]		# ADDED
			
			# For each training X:
			for i in range(len(X)):
				x = X[i]
				M = len(X[i])
				# Compute alphas and betas matrices:
				alphas = self.forward(x, normalize=True)
				betas = self.backward(x, normalize=True)

				# For index of each element in x:
				for j in range(1, M+1): # ignore j=0 since we get marginal prob = 0
					norm_O = np.dot(alphas[j], betas[j])
					norm_A = 0.0
					temp_A = [[0 for _ in range(self.L)] for _ in range(self.L)]

					for state1 in range(self.L):
						if norm_O == 0:
							continue
						numerator_O[state1][x[j-1]] += alphas[j][state1] * betas[j][state1] / norm_O
						denom_O[state1][x[j-1]] += alphas[j][state1] * betas[j][state1] / norm_O	# ADDED

						if j < M:
							for state2 in range(self.L):
								temp_A[state1][state2] += alphas[j][state1] * self.O[state2][x[j]] * self.A[state1][state2] * betas[j+1][state2]
								norm_A += alphas[j][state1] * self.O[state2][x[j]] * self.A[state1][state2] * betas[j+1][state2]
								if norm_O == 0:
									continue
								denom_A[state1][state2] += alphas[j][state1] * betas[j][state1] / norm_O	# ADDED
					if j < M:
						if norm_A == 0:
							temp_A = np.array(temp_A)
							numerator_A = list(np.array(numerator_A) + temp_A)
						else:
							temp_A = np.array(temp_A) / norm_A
							numerator_A = list(np.array(numerator_A) + temp_A)
			for i in range(self.L):
				for b in range(self.L):
					self.A[i][b] = numerator_A[i][b] / denom_A[i][b]
				for j in range(self.D):
					self.O[i][j] = numerator_O[i][j] / np.sum(numerator_O[i])


	def generate_emission(self, M):
		'''
		Generates an emission of length M, assuming that the starting state
		is chosen uniformly at random. 

		Arguments:
			M:          Length of the emission to generate.

		Returns:
			emission:   The randomly generated emission as a list.

			states:     The randomly generated states as a list.
		'''
		emission = []
		states = []
		start_state = random.choice(range(self.L))
		curr_state = start_state

		for i in range(M):
			# Sample the state using transition matrix
			rand = random.uniform(0, 1)
			for j in range(len(self.A[curr_state])):
				rand -= self.A[curr_state][j]
				if rand < 0:
					curr_state = j
					states.append(curr_state)
					break


			# Sample the emission using observation matrix
			rand = random.uniform(0, 1)
			for j in range(len(self.O[curr_state])):
				rand -= self.O[curr_state][j]
				if rand < 0:
					emission.append(j)
					break

		return emission, states


	def probability_alphas(self, x):
		'''
		Finds the maximum probability of a given input sequence using
		the forward algorithm.

		Arguments:
			x:          Input sequence in the form of a list of length M,
						consisting of integers ranging from 0 to D - 1.

		Returns:
			prob:       Total probability that x can occur.
		'''

		# Calculate alpha vectors.
		alphas = self.forward(x)

		# alpha_j(M) gives the probability that the state sequence ends
		# in j. Summing this value over all possible states j gives the
		# total probability of x paired with any state sequence, i.e.
		# the probability of x.
		prob = sum(alphas[-1])
		return prob


	def probability_betas(self, x):
		'''
		Finds the maximum probability of a given input sequence using
		the backward algorithm.

		Arguments:
			x:          Input sequence in the form of a list of length M,
						consisting of integers ranging from 0 to D - 1.

		Returns:
			prob:       Total probability that x can occur.
		'''

		betas = self.backward(x)

		# beta_j(1) gives the probability that the state sequence starts
		# with j. Summing this, multiplied by the starting transition
		# probability and the observation probability, over all states
		# gives the total probability of x paired with any state
		# sequence, i.e. the probability of x.
		prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
					for j in range(self.L)])

		return prob


def supervised_HMM(X, Y):
	'''
	Helper function to train a supervised HMM. The function determines the
	number of unique states and observations in the given data, initializes
	the transition and observation matrices, creates the HMM, and then runs
	the training function for supervised learning.

	Arguments:
		X:          A dataset consisting of input sequences in the form
					of lists of variable length, consisting of integers 
					ranging from 0 to D - 1. In other words, a list of lists.

		Y:          A dataset consisting of state sequences in the form
					of lists of variable length, consisting of integers 
					ranging from 0 to L - 1. In other words, a list of lists.
					Note that the elements in X line up with those in Y.
	'''
	# Make a set of observations.
	observations = set()
	for x in X:
		observations |= set(x)

	# Make a set of states.
	states = set()
	for y in Y:
		states |= set(y)
	
	# Compute L and D.
	L = len(states)
	D = len(observations)

	# Randomly initialize and normalize matrix A.
	A = [[random.random() for i in range(L)] for j in range(L)]

	for i in range(len(A)):
		norm = sum(A[i])
		for j in range(len(A[i])):
			A[i][j] /= norm
	
	# Randomly initialize and normalize matrix O.
	O = [[random.random() for i in range(D)] for j in range(L)]

	for i in range(len(O)):
		norm = sum(O[i])
		for j in range(len(O[i])):
			O[i][j] /= norm

	# Train an HMM with labeled data.
	HMM = HiddenMarkovModel(A, O)
	HMM.supervised_learning(X, Y)

	return HMM

def unsupervised_HMM(X, n_states, N_iters):
	'''
	Helper function to train an unsupervised HMM. The function determines the
	number of unique observations in the given data, initializes
	the transition and observation matrices, creates the HMM, and then runs
	the training function for unsupervised learing.

	Arguments:
		X:          A dataset consisting of input sequences in the form
					of lists of variable length, consisting of integers 
					ranging from 0 to D - 1. In other words, a list of lists.

		n_states:   Number of hidden states to use in training.
		
		N_iters:    The number of iterations to train on.
	'''
	# random.seed(2019)

	# Make a set of observations.
	observations = set()
	for x in X:
		observations |= set(x)
	
	# Compute L and D.
	L = n_states
	D = len(observations)

	# Randomly initialize and normalize matrix A.
	A = [[random.random() for i in range(L)] for j in range(L)]

	for i in range(len(A)):
		norm = sum(A[i])
		for j in range(len(A[i])):
			A[i][j] /= norm
	
	# Randomly initialize and normalize matrix O.
	O = [[random.random() for i in range(D)] for j in range(L)]

	for i in range(len(O)):
		norm = sum(O[i])
		for j in range(len(O[i])):
			O[i][j] /= norm

	# Train an HMM with unlabeled data.
	HMM = HiddenMarkovModel(A, O)
	HMM.unsupervised_learning(X, N_iters)

	return HMM