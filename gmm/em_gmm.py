import numpy as np
from scipy.stats import multivariate_normal as mn
from random import random
import matplotlib.pyplot as plt


class em_gmm():
    def __init__(self, data, K, centroids):
        self.data = data
        self.K = K
        self.n_k = np.zeros((1, K))
        self.pi = np.array([1 / K for i in range(K)])
        self.phi = np.zeros((len(data), K))
        self.mean = centroids
        self.sigma = [np.identity(data.shape[1]) for i in range(K)]

        self.update_phi()

    def train(self, N):
        for n in range(N):
            # E STEP
            self.update_phi()

            # M STEP
            self.update_n_k()
            self.update_pi()
            self.update_mean()
            self.update_sigma()

            # save
            for k in range(self.K):
                k += 1
                file = "Sigma-" + str(k) + "-" + str(n+1) + ".csv"
                np.savetxt(file, self.sigma[k - 1], delimiter=",")
            file = "pi-" + str(n+1) + ".csv"
            np.savetxt(file, self.pi, delimiter=",")
            file = "mu-" + str(n+1) + ".csv"
            np.savetxt(file, self.mean, delimiter=",")

    def update_phi(self):
        for d in range(len(self.data)):
            total = 0
            for k in range(self.K):
                total += self.pi[k] * mn.pdf(self.data[d], mean=self.mean[k], cov=self.sigma[k], allow_singular=True)

            for k in range(self.K):
                self.phi[d][k] = self.pi[k] * mn.pdf(self.data[d], mean=self.mean[k], cov=self.sigma[k], allow_singular=True) / total

    def update_n_k(self):
        self.n_k = np.sum(self.phi, axis=0)
        self.n_k = np.array(self.n_k)

    def update_pi(self):
        for k in range(self.K):
            self.pi[k] = self.n_k[k] / len(self.data)

    def update_mean(self):
        self.mean = np.zeros((self.K, self.data.shape[1]))
        for k in range(self.K):
            for d in range(len(self.data)):
                if self.n_k[k] == 0:
                    print("here:", self.n_k)
                    print(self.phi)
                self.mean[k] += self.phi[d][k] * self.data[d] / self.n_k[k]

    def update_sigma(self):
        self.sigma = [np.zeros((self.data.shape[1], self.data.shape[1])) for i in range(self.K)]
        for k in range(self.K):
            for d in range(len(self.data)):
                self.sigma[k] += self.phi[d][k] / self.n_k[k] * np.multiply(
                    self.data[d] - self.mean[k],
                    (self.data[d] - self.mean[k])[np.newaxis].T
                )


if __name__ == "__main__":
    # create graph points for basic testing
    centroids = [[0, 0], [15, 15]]
    xs = []
    ys = []
    for i in range(50):
        for c in centroids:
            xs.append(c[0] + random() * 5 - random() * 5)
            ys.append(c[1] + random() * 5 - random() * 5)

    # create initial guess
    guess = []
    for c in centroids:
        guess.append([
            c[0] + 5 * (random() - random()),
            c[1] + 5 * (random() - random())
        ])

    # construct data
    data = []
    for i in range(len(xs)):
        data.append([xs[i], ys[i]])

    alg = em_gmm(np.array(data), 2, np.array(guess))
    alg.train(10)

    plt.plot(xs, ys, "o", color="black")
    for g in guess:
        plt.plot(g[0], g[1], "x", color="red")

    for m in alg.mean:
        plt.plot(m[0], m[1], "x", color="green")
        print(m[0], m[1])

    plt.show()