import numpy as np
import sys


# probabilistic matrix factorizer
class PMF():
    def __init__(self, data):
        self.lam = 2        # lambda - penalty constant
        self.sigma2 = 0.1   # variance
        self.d = 5          # number of features to learn
        self.sets = []      # all (user, object) indices of M that have ratings
        self.i = []         # contains a list of objects rated for each user
        self.j = []         # contains a list of users that rated each object

        # matrix factorization M ~ U V
        self.M = self.create_M(data)
        self.U = np.random.rand(self.M.shape[0], self.d) / self.lam
        self.V = np.random.rand(self.d, self.M.shape[1]) / self.lam

    # create a matrix based on data where each row has the format:
    # [user_index, object_index, rating]
    def create_M(self, data):
        # find the total number of users and objects
        users = int(np.max(data[:, 0]))
        objects = int(np.max(data[:, 1]))

        # create bins to store objects rated by each user and users that rated each object
        self.i = [[] for i in range(users)]
        self.j = [[] for j in range(objects)]

        # initialize the incomplete matrix with zeros
        M = np.zeros((users, objects))

        # go through the data file and add the data to matrix M
        for i in range(data.shape[0]):
            # convert the user and object indices to indices for M
            u = int(data[i][0]) - 1
            o = int(data[i][1]) - 1

            # add each rating to M
            M[u][o] = data[i][2]

            # add indices to bins
            self.i[u].append(o)
            self.j[o].append(u)
            self.sets.append([u, o])

        return M

    # train the data using N iterations of PMF
    def train(self, N):
        # bin to hold all objective calculations from each iteration
        obj = []

        # iterate the algorithm N times
        for n in range(N):
            # update the feature matrices
            self.update_U()
            self.update_V()

            # calculate the objective function
            obj.append([self.calc_objective()])

            # if this is the 10th, 25th, or 50th iteration, save feature matrices to a csv file
            if n + 1 in [10, 25, 50]:
                file_u = "U-" + str(n + 1) + ".csv"
                file_v = "V-" + str(n + 1) + ".csv"
                np.savetxt(file_u, self.U, delimiter=",")
                np.savetxt(file_v, self.V.T, delimiter=",")

        # print statements to verify the program has finished
        print("U:", self.U.shape)
        print("V:", self.V.shape)

        # save the objective function calculations to a csv file
        np.savetxt("objective.csv", np.array(obj), delimiter=",")

    # update the user feature matrix
    def update_U(self):
        # bin to hold updated user features vectors
        us = []

        # penalty matrix
        penalty = self.lam * self.sigma2 * np.identity(self.d)

        # for each user
        for i in range(self.M.shape[0]):
            sum1 = np.zeros(self.d, dtype=float)
            sum2 = np.array([0 for i in range(self.d)], dtype=np.float)

            # for each object rated by the user
            for j in self.i[i]:
                v = self.V[:, j]
                sum1 = sum1 + np.multiply(v, v[np.newaxis].T)
                sum2 = sum2 + self.M[i][j] * v

            # compute the update for the user's feature vector
            prod = np.linalg.inv(penalty + sum1)
            us.append(np.dot(prod, sum2))

        # update the user matrix
        self.U = np.array(us)

    # update the object feature matrix
    def update_V(self):
        # bin to hold object feature vectors
        vs = []

        # penalty matrix
        penalty = self.lam * self.sigma2 * np.identity(self.d)

        # for each object
        for j in range(self.M.shape[1]):
            sum1 = np.zeros(self.d, dtype=float)
            sum2 = np.array([0 for i in range(self.d)], dtype=float)

            # for each user that rated the object
            for i in self.j[j]:
                u = self.U[i, :]
                sum1 = sum1 + np.multiply(u, u[np.newaxis].T)
                sum2 = sum2 + self.M[i][j] * u

            # compute the update for the object's features
            prod = np.linalg.inv(penalty + sum1)
            vs.append(np.dot(prod, sum2))

        # update the object matrix
        self.V = np.array(vs).T

    # calculate the objective function
    def calc_objective(self):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        V = self.V.T
        for s in self.sets:
            # sum1 += np.linalg.norm(self.M[s[0]][s[1]] - np.multiply(self.U[s[0]], V[s[1]]))
            sum1 += np.linalg.norm(self.M[s[0]][s[1]] - np.dot(self.U[s[0]], V[s[1]][np.newaxis].T))
        for u in self.U:
            sum2 += np.linalg.norm(u)
        for v in V:
            sum3 += np.linalg.norm(v)

        return - 1 / 2 / self.sigma2 * sum1 - self.lam / 2 * (sum2 + sum3)


if __name__ == "__main__":
    data = np.genfromtxt("ratings_sample.csv", delimiter=",")
    # data = np.genfromtxt(sys.argv[1], delimiter = ",")
    pmf = PMF(data)
    pmf.train(5)
