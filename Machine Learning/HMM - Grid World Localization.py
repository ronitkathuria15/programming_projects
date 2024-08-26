import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: bool = False):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1
        else:
            self.grid = np.random.randint(2, size=size)

        self.epsilon = epsilon
        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        M, N = self.grid.shape
        adjacent = [
            (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), 
            (i, j),(i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1),
        ]
        neighbors = []
        for a in adjacent:
            if a[0] >= 0 and a[0] < M and a[1] >= 0 and a[1] < N and self.grid[a] == 0:
                neighbors.append(a)
        return neighbors

    """
    4.1 Transition and observation probabilities
    """

    def initT(self):
        """
        Create and return NxN transition matrix, where N = size of grid.
        """
        M, N = self.grid.shape
        T = np.zeros((M * N, M * N))
        for i in range(M):
            for j in range(N):
                
              #flattening indices
              given_state = i * N + j
              neighbors = self.neighbors((i,j))
              num_neighbors = len(neighbors)

              #checking for blocked cells
              if (num_neighbors == 0):
                  T[given_state][given_state] = 1
              
              else:
              #setting probabilities of transition matrix for free cells
                for n in neighbors:
                  adjacent_state = n[0] * N + n[1]
                  T[adjacent_state][given_state] = 1/num_neighbors
                      
              #normalizing
        for j in range(M*N): 
          colSum = np.sum(T[:, j])
          T[:, j] /= colSum
        return T

    def initO(self):
        """
        Create and return 16xN matrix of observation probabilities, where N = size of grid.
        """
        M, N = self.grid.shape
        O = np.zeros((16, M * N))
        #determining correct observation 
        for j in range(N):
          for i in range(M):
                n = self.neighbors((i,j))
                correctObservation = 0
                #north
                if (i-1,j) not in n: 
                  correctObservation += 8
                #south
                if (i+1,j) not in n:  
                  correctObservation += 2
                #east
                if (i,j+1) not in n:
                  correctObservation += 4
                #west 
                if (i,j-1) not in n:
                  correctObservation += 1

                #computing the discrepancy and observation probability 
                for observations in range(16):
                  discrepancy = bin(observations ^ correctObservation)
                  discrepancy = discrepancy.count('1')
                  O[observations][i * N + j] = ((1 - self.epsilon)**(4 - discrepancy))*(self.epsilon ** discrepancy)

        return O

    """
    4.2 Inference: Forward, backward, filtering, smoothing
    """

    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        alpha_next = np.diag(self.obs[observation]) @ (self.trans @ alpha)
        return alpha_next

    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current "message" of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated message.
        """
        beta_prime = self.obs[observation] * beta
        beta_first = self.trans.T @ beta_prime
        return beta_first

    def filtering(self, init: npt.ArrayLike, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Estimated belief state at each timestep.
        """
        #creating belief state array
        belief_states = np.zeros((self.grid.size, len(observations)))
        #determining initial belief state 
        belief_states[:, 0] = self.forward(init, observations[0])
        #determining belief state at each time t
        for t in range(1, len(observations)):
           belief_states[:, t] = self.forward(belief_states[:,t-1], observations[t])
           #normalizing
           colSum = np.sum(belief_states[:, t])
           belief_states[:, t] /= colSum
        return belief_states
        
    def smoothing(self, init: npt.ArrayLike, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Smoothed belief state at each timestep.
        """
        #calling filtering to determine alpha probabilities 
        alpha = self.filtering(init, observations)
        #creating beta array 
        beta = np.zeros((self.grid.size, len(observations)))
        #determining initial beta 
        beta[:, len(observations) - 1] = self.backward(init, observations[len(observations) - 1])
        #determining beta for each time t
        for t in range(len(observations) - 2, -1, -1):
           beta[:, t] = self.backward(beta[:, t + 1], observations[t + 1])
           #normalizing
           colSum = np.sum(beta[:, t])
           beta[:, t] /= colSum
        #taking Hadamard product of alpha and beta 
        smoothed = alpha * beta 
        #normalizing
        a,b = smoothed.shape 
        for t in range(b):
          colSum = np.sum(smoothed[:, t])
          smoothed[:, t] /= colSum
        
        return smoothed

    """
    4.3 Localization error
    """

    def loc_error(self, beliefs: npt.ArrayLike, trajectory: list[int]):
        """Compute localization error at each timestep.
        Args:
          beliefs (np.ndarray): Belief state at each timestep.
          trajectory (list[int]): List of states visited.
        Returns:
          list[int]: Localization error at each timestep.
        """
        a,b = beliefs.shape
        M,N = self.grid.shape
       #determining predicted states
        predicted = np.argmax(beliefs, axis = 0)
        #creating errror list
        errors = np.zeros(len(trajectory))
        #determining manhattan error
        for i in range(b):
           p = trajectory[i] // N
           q = trajectory[i] % N

           r = predicted[i] // N
           s = predicted[i] % N
           errors[i] = np.abs(r - p) + np.abs(s - q)
        return errors 
    
        