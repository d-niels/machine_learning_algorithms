import numpy as np
import sys
from kmeans import k_means
from em_gmm import em_gmm

if __name__ == '__main__':
    K = 3
    data = np.genfromtxt("iris.data", delimiter=',')
    #data = np.genfromtxt(sys.argv[1], delimiter=",")
    centroids = k_means(data, K)
    em = em_gmm(data, K, centroids)
    em.train(10)