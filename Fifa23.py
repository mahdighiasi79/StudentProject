import copy
import pandas as pd
import numpy as np
import pickle


categorical_values = ["Known As", "Full Name", "Positions Played", "Best Position", "Nationality", "Image Link",
                      "Club Name", "Club Position", "On Loan", "Preferred Foot", "National Team Name",
                      "National Team Image Link", "National Team Position", "National Team Jersey Number",
                      "Attacking Work Rate", "Defensive Work Rate", "Contract Until", "Joined On", "Club Jersey Number"]


def extract_numerical_values():
    df = pd.read_csv("Fifa 23 Players Data.csv")
    columns = df.columns
    numerical_values = []

    for i in range(len(df)):
        numerical_values.append([])
        for j in range(len(columns)):
            if columns[j] not in categorical_values:
                numerical_values[i].append(df.iloc(0)[i][j])

    with open("numerical values.bin", "wb") as file:
        pickle.dump(numerical_values, file)


def covariance_matrix(input_matrix):
    n = len(input_matrix)
    d = len(input_matrix[0])
    sigmoid_matrix = np.zeros((d, d))
    for i in range(n):
        x_i = np.reshape(copy.deepcopy(input_matrix[i]), (d, 1))
        x_i_t = np.reshape(copy.deepcopy(input_matrix[i]), (1, d))
        sigmoid_matrix += x_i @ x_i_t
    sigmoid_matrix /= n
    return sigmoid_matrix


def pca(input_matrix):
    sigmoid_matrix = covariance_matrix(input_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(sigmoid_matrix)
    sorted_eigenvalues = sorted(eigenvalues, reverse=True)

    new_basis = []
    for i in range(len(eigenvalues)):
        if eigenvalues[i] == sorted_eigenvalues[0]:
            new_basis.append(eigenvectors[i])
            break
    for i in range(len(eigenvalues)):
        if eigenvalues[i] == sorted_eigenvalues[1]:
            new_basis.append(eigenvectors[i])
            break
    new_basis = np.array(new_basis)

    result = copy.deepcopy(input_matrix)
    result = new_basis @ result.transpose()
    return result.transpose()


def k_means(points, k, iterations):
    n, d = points.shape
    centers = np.random.random((k, d))

    for c in range(iterations):
        members = [[]] * k
        for i in range(n):
            nearest_center = -1
            nearest_distance = np.inf
            for j in range(k):
                distance = np.linalg.norm(points[i] - centers[j])
                if distance < nearest_distance:
                    nearest_center = j
                    nearest_distance = distance
            members[nearest_center].append(i)
        members = np.array(members)

        for i in range(k):
            cluster = members[i]
            new_center = np.zeros(d)
            for element in cluster:
                new_center += element
            new_center /= len(cluster)
            centers[i] = new_center

    return centers


def clustering(points, centers):
    clusterings = []
    for i in range(len(points)):
        nearest_center = -1
        nearest_distance = np.inf
        for j in range(len(centers)):
            distance = np.linalg.norm(points[i] - centers[j])
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_center = j
        clusterings[i] = nearest_center
    return clusterings


if __name__ == "__main__":
    #extract_numerical_values()

    with open("numerical values.bin", "rb") as f:
        nv = pickle.load(f)
        f.close()
    best_players = np.array(nv[:91])

    a = [[]] * 3
    print(a)
