from numpy import *


def load_dataset():
    data_matrix = []
    label_matrix = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        tokens = line.strip().split()
        data_matrix.append([1.0, float(tokens[0]), float(tokens[1])])
        label_matrix.append(int(tokens[2]))
    return mat(data_matrix), mat(label_matrix).transpose()


def sigmod(input):
    return 1.0 / (1 + exp(-input))


def gradient_ascent(data_matrix, label_matrix):
    m, n = shape(data_matrix)
    alpha = 0.001
    num_cycle = 500
    weight_vector = ones((n, 1))
    for k in range(num_cycle):
        sigmod_val = sigmod(data_matrix * weight_vector)
        prediction_error = label_matrix - sigmod_val
        weight_vector = weight_vector + alpha * data_matrix.transpose() * prediction_error
    return weight_vector
