import numpy as np


# sorted list that keeps the 10 highest values
# values are stored in pairs, the first is the index and the second is the value being compared
class SortedList():
    def __init__(self, n=10):
        self.n = n
        self.max_list = [[0, -1e9] for i in range(n)]

    # add a value if it is bigger than another value in the list
    def add(self, value):
        # check each value in the list
        for i in range(len(self.max_list)):
            if value[1] > self.max_list[i][1]:
                # add the value to this position and push everything after it
                # back one position
                j = self.n - 2
                while j >= i:
                    self.max_list[j + 1][0] = self.max_list[j][0]
                    self.max_list[j + 1][1] = self.max_list[j][1]
                    j = j - 1
                self.max_list[i][0] = value[0]
                self.max_list[i][1] = value[1]
                break

    # print the list and change any values that haven't been changed to '-'
    def print(self):
        print_list = []
        for i in range(len(self.max_list)):
            # find values that haven't been changed
            if self.max_list[i] == -1e9:
                print_list.append('-')
            else:
                print_list.append(self.max_list[i])
        print(print_list)

    # returns the list of pairs
    def get_list(self):
        return self.max_list

    # returns the list of indexes
    def get_xs(self):
        xs = []
        for x in self.max_list:
            xs.append(x[0])
        return xs


# L2-regularized least squares linear regression
# Input: lambda, matrix of input vectors, vector of output values
# Return: wRR
def l2_least_squares(lam, X_train, y_train):

    # Get dimension of X
    X_size = X_train.shape[1]

    # w = (lambda * I + X^T * X)^-1 * X^T * y
    return np.linalg.multi_dot([np.linalg.inv(lam * np.identity(X_size) + np.dot(X_train.T, X_train)), X_train.T, y_train])


# Input: lambda, sigma, matrix of input vectors, matrix of tests vectors
# Return: list of ten best tests values for wRR
def active_learning(lam, sig2, X_train, X_test):

    # Get dimension of X_train
    X_size = X_train.shape[1]

    # find squared value of all vectors of X_train
    sqrd_sum = 0
    for x in X_train:
        sqrd_sum += np.dot(x, x.T)

    # create sorted list
    best = SortedList()

    # i is the index of the tests vector
    i = 1
    for x0 in X_test:
        # calculate covariance matrix with new added data point
        # (lam * I + sigma^-2 * X^T * X)^-1
        cov = np.linalg.inv(lam * np.identity(X_size) + sig2**-1 * np.dot(X_train.T, X_train))

        # best data to use maximizes this function:
        # sigma^2 + x0^T * cov * x0
        best.add([i, sig2 + np.linalg.multi_dot([x0.T, cov, x0])])
        i += 1

    # return the 10 best values
    return best.get_xs()


# Get inputs
lambda_input = 2                                            # int(sys.argv[1])
sigma2_input = 2                                            # float(sys.argv[2])
X_train = np.genfromtxt('X_train.csv', delimiter=',')       # np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt('y_train.csv')                      # np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt('X_test.csv', delimiter=',')         # np.genfromtxt(sys.argv[5], delimiter = ",")

# Compute Solutions
wRR = l2_least_squares(lambda_input, X_train, y_train)
active = active_learning(lambda_input, sigma2_input, X_train, X_test)

# write output to file
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n")
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",")