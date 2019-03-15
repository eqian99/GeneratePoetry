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
import numpy as np

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
        self.Rhyme_map = {}
        self.Rhyme_counter = 0


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

        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]
            seqs[1][i] = str(i)

        for i in range(2, M + 1):
            for k in range(self.L):
                max_prob = 0
                max_seq = ''
                for j in range(self.L):
                    # Find probability of state k given that the previous state was j
                    probability = probs[i - 1][j] * self.A[j][k] * self.O[k][x[i - 1]]
                    if (probability > max_prob):
                        max_prob = probability
                        max_seq = seqs[i - 1][j] + str(k)
                probs[i][k] = max_prob
                seqs[i][k] = max_seq

        m = max(probs[M])
        max_probability = [i for i, j in enumerate(probs[M]) if j == m]
        return seqs[M][max_probability[0]]


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

        for z in range(self.L):
            alphas[0][z] = self.A_start[z]
            alphas[1][z] = self.O[z][x[0]] * self.A_start[z]

        for i in range(2, M + 1):
            for z in range(self.L):
                sum = 0
                for j in range (self.L):
                    sum += alphas[i - 1][j] * self.A[j][z]
                alphas[i][z] = self.O[z][x[i - 1]] * sum

            if normalize:
                alphas[i] = (np.array(alphas[i]) / np.sum(alphas[i])).tolist()

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
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for z in range(self.L):
            betas[M][z] = 1

        for i in range(M - 1, -1, -1):
            for z in range(self.L):
                betas[i][z] = 0
                for j in range(self.L):
                    betas[i][z] += betas[i + 1][j] * self.A[z][j] * self.O[j][x[i]]

            if normalize:
                betas[i] = (np.array(betas[i]) / np.sum(betas[i])).tolist()

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

        L = self.L
        D = self.D

        A = [[0] * (L) for i in range(L)]
        O = [[0] * (D) for i in range(L)]

        for i in range(len(Y)):
            for j in range(len(Y[i])):
                val = Y[i][j]
                if (j > 0):
                    old_val = Y[i][j - 1]
                    A[old_val][val] += 1

        for i in range(L):
            norm = 0
            for j in range(L):
                norm += A[i][j]
            for j in range(L):
                A[i][j] /= norm

        # Calculate each element of O using the M-step formulas.

        for i in range(len(X)):
            for j in range(len(X[i])):
                input = X[i][j]
                state = Y[i][j]
                O[state][input] += 1

        for i in range(L):
            norm = 0
            for j in range(D):
                norm += O[i][j]
            for j in range(D):
                O[i][j] /= norm


        self.A = A
        self.O = O


    def unsupervised_learning(self, X, N_iters, Rhyme_map, Rhyme_counter):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        L = self.L
        D = self.D
        self.Rhyme_map = Rhyme_map
        self.Rhyme_counter = Rhyme_counter

        for n in range(N_iters):
            An = [[0] * (L) for i in range(L)]
            On = [[0] * (D) for i in range(L)]
            Ad = [0] * (L)
            Od = [0] * (D)
            for l in range(len(X)):
                M = len(X[l])
                alphas = self.forward(X[l], normalize=True)
                betas = self.backward(X[l], normalize=True)

                OM = [[0] * (L) for i in range(M + 1)]
                OMN = [[0] * (L) for i in range(M + 1)]
                OMD = [0] * (M + 1)
                AM = [[[0] * (L) for i in range(L)] for j in range(M + 1)]
                AMN = [[[0] * (L) for i in range(L)] for j in range(M + 1)]
                AMD = [0] * (M + 1)

                for j in range(1, M + 1):
                    for z in range(L):
                        OMN[j][z] += alphas[j][z] * betas[j][z]
                        OMD[j] += alphas[j][z] * betas[j][z]

                for j in range(1, M + 1):
                    for z in range(L):
                        OM[j][z]  = (OMN[j][z]) / OMD[j]

                for j in range(1, M):
                    for a in range(L):
                        for b in range(L):
                            AMN[j][a][b] = alphas[j][a] * self.O[b][X[l][j]] * self.A[a][b] * betas[j + 1][b]
                            AMD[j] += alphas[j][a] * self.O[b][X[l][j]] * self.A[a][b] * betas[j + 1][b]

                for i in range(1, M):
                    for a in range(L):
                        for b in range(L):
                            AM[i][a][b] = (np.array(AMN[i][a][b]) / AMD[i]).tolist()

                for a in range(L):
                    for b in range(L):
                        for i in range(1, M):
                            An[a][b] += AM[i][a][b]
                    for k in range(1, M):
                        Ad[a] += OM[k][a]

                for a in range(L):
                    for i in range(D):
                        for j in range(0, M):
                            if (X[l][j] == i):
                                On[a][i] += OM[j + 1][a]
                    for k in range(0, M):
                        Od[a] += OM[k + 1][a]

            for a in range(L):
                for b in range(L):
                    self.A[a][b] = (np.array(An[a][b]) / Ad[a]).tolist()
                for b in range(D):
                    self.O[a][b] = (np.array(On[a][b]) / Od[a]).tolist()

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
        rhymes = np.zeros(14)
        rhyme_list = np.random.randint(self.Rhyme_counter, size=7)
        rhyme_indices = [1, 2, 8, 9, 3, 4, 10, 11, 5, 6, 12, 13, 7, 14]

        L = self.L
        D = self.D
        for j in range(len(rhyme_indices)):
            for i in range(len(rhyme_list)):
                part1, part2 = self.Rhyme_map[rhyme_list[i]]
                if (i == rhyme_indices[j]):
                    rhymes[j] = part1
                elif (i == rhyme_indices[j] - 7):
                    rhymes[j] = part2
                    
        for i in range(M):
            if (i == 0):
                # TODO Generate the 14 rhyming words at the end
                # TODO Initialize state to most likely for chosen word
                state = np.random.randint(L)
                states.append(state)
                probs = self.O[state]
                # TODO Don't need to choose an emission here.  Already chosen
                emission_choice = np.random.choice(list(range(D)), p=probs)
                emission.append(emission_choice)
            else:
                probs = self.A[:, states[-1]]
                state = np.random.choice(list(range(L)), p=probs)
                states.append(state)
                probs = self.O[state]
                emission_choice = np.random.choice(list(range(D)), p=probs)
                emission.append(emission_choice)

        return emission.reverse(), states.reverse()


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

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)
    random.seed(2019)

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
