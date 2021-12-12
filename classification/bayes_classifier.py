from __future__ import division
import numpy as np
from sklearn.covariance import EmpiricalCovariance
import sys


# Bayes plugin classifier
class PluginClassifier(object):
    def __init__(self, data, ys, n=10):
        self.n = n
        self.ys = ys
        self.data = data
        self.y_counts = self.get_y_counts()
        self.means = self.get_means()
        self.covs = self.get_covs()
        print(self.covs)
        print(self.y_counts)
        print(self.means)

    # counts the number of occurrences of each y in the data
    def get_y_counts(self):
        y_counts = [0 for i in range(self.n)]
        for y in self.ys:
            y_counts[y] += 1

        return y_counts

    # calculates the mean for each y
    def get_means(self):
        means = [0 for i in range(self.n)]
        for i in range(len(self.data)):
            means[self.ys[i]] += self.data[i] / self.y_counts[self.ys[i]]

        return means

    # calculates the covariance matrix for each y
    def get_covs(self):
        covs = [0 for i in range(self.n)]
        for i in range(len(self.data)):
            sub = np.subtract(self.data[i], self.means[self.ys[i]])[np.newaxis]
            covs[self.ys[i]] += np.multiply(sub.T, sub) / self.y_counts[self.ys[i]]

        return covs

    # predict a class for vector X
    def predict(self, X):
        probs = []
        for y in range(self.n):
            sub = np.subtract(X, self.means[y])[np.newaxis]
            p = self.y_counts[y] / len(self.ys)
            p = p * np.linalg.det(self.covs[y]) ** -.5
            p = p * np.e**(
                -.5 * np.linalg.multi_dot([sub, np.linalg.inv(self.covs[y]), sub.T])
            )

            probs.append(p[0][0])

        probs = np.array(probs, dtype=float)

        # normalize probabilities
        probs = probs / sum(probs)

        return probs


# main function
def main():
    """
    # global constants
    X_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2]).astype(int)
    X_test = np.genfromtxt(sys.argv[3], delimiter=",")
    """


    X_train = np.genfromtxt("iris.data", delimiter=",", usecols=(0, 1, 2, 3))
    y_train = np.genfromtxt("iris.data", delimiter=",", usecols=(4), dtype=int)
    X_test = X_train

    # classifier object with training data
    classifier = PluginClassifier(X_train, y_train, n=3)

    # classifies the test data using the model trained
    predictions = []
    for x in X_test:
        predictions.append(classifier.predict(x))

    final_outputs = predictions
    np.savetxt("probs_test.csv", final_outputs, delimiter=",")


if __name__ == "__main__":
    main()
