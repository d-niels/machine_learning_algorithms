import numpy as np
from random import random

# iterations of k means
N = 10


# create k random centroids within the limits of the data
def initialize_centroids(data, k):
    # get bounds
    high = np.amax(data, axis=0)
    low = np.amin(data, axis=0)

    # create random points within the bounds
    centroids = []
    for i in range(k):
        centroid = []
        for j in range(high.shape[0]):
            centroid.append(random() * (high[j] - low[j]) + low[j])
        centroids.append(centroid)
    centroids = np.array(centroids)

    return centroids


# return the closest centroid
def find_cluster(vec, centroids):
    best_d = 1e6
    best_i = 1e6
    for i in range(centroids.shape[0]):
        d = np.linalg.norm(vec - centroids[i])
        if d < best_d:
            best_i = i
            best_d = d

    return best_i


# calculate the centroid of each cluster
def calculate_centroids(data, assignments, k):
    # flag to keep track if the program converged to less than k clusters
    err = False

    centroids = np.zeros((k, data.shape[1]))
    c_count = [0 for i in range(k)]

    # add up all data vectors who were assigned to each cluster
    for d in range(data.shape[0]):
        centroids[assignments[d]] += data[d]
        c_count[assignments[d]] += 1

    # divide by the number of the vectors assigned to each cluster
    for c in range(len(c_count)):
        if c_count[c] == 0:
            err = True
        centroids[c] = centroids[c] / c_count[c]

    return centroids, err


# k_means algorithm
def k_means(data, k):
    # flag to rerun if program converges to less than k clusters
    rerun = False

    # keeps track of the cluster each data point belongs to
    assignments = [0 for i in range(data.shape[0])]

    # keeps track of the centroids
    centroids = initialize_centroids(data, k)

    # do N iterations
    for n in range(N):
        # find which cluster each point belongs to
        for d in range(data.shape[0]):
            assignments[d] = find_cluster(data[d], centroids)

        # find the centroids of the new clusters
        centroids, err = calculate_centroids(data, assignments, k)

        # check of there was a divide by 0
        if err:
            rerun = True
            break

        # save the centroids to a file
        n += 1
        file = "centroids-" + str(n) + ".csv"
        np.savetxt(file, centroids, delimiter=",")

    if rerun:
        return k_means(data, k)
    else:
        return centroids.copy()


if __name__ == "__main__":
    data = np.genfromtxt("iris.data", delimiter=',')
    k = 3
    k_means(data, k)
