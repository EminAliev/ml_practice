import numpy as np
import matplotlib.pyplot as plt


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def cluster(matrix, n, k):
    clust = np.zeros(n)
    for i in range(n):
        max_affiliation = max(matrix[i])
        for j in range(k):
            if matrix[i][j] == max_affiliation:
                clust[i] = j
    return clust


def draw(x, y, clust, x_cc, y_cc):
    colors = ['b', 'g', 'r', 'darkmagenta', 'm', 'y', 'k', 'w']
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=colors[int(clust[i])])

    plt.scatter(x_cc, y_cc, marker='X')
    plt.show()


def recntr(x, y, matrix, k, m):
    x_cc = np.zeros(k)
    y_cc = np.zeros(k)

    for i in range(k):
        n = 0
        sum_x = 0
        sum_y = 0

        for j in range(len(x)):
            max = 0
            for t in matrix[j]:
                if t > max:
                    max = t
            if matrix[j, i] == max:
                n = n + matrix[j, i] ** m
                sum_x = sum_x + matrix[j, i] ** m * x[j]
                sum_y = sum_y + matrix[j, i] ** m * y[j]

        if n != 0:
            x_cc[i] = sum_x / n
            y_cc[i] = sum_y / n
        else:
            x_cc[i] = 0
            y_cc[i] = 0

    return x_cc, y_cc


def calculate_new_probability_clusters(n, k, x, y, x_c, y_c, m):
    matrix = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            sum = 0
            distance_to_center_j = dist(x[i], y[i], x_c[j], y_c[j])
            for t in range(k):
                distance_to_center_t = dist(x[i], y[i], x_c[t], y_c[t])
                sum += (distance_to_center_j / distance_to_center_t) ** (2 / (m - 1))
            matrix[i, j] = 1 / sum
    return matrix


def did_algo_converge(old_matrix, matrix, n, k, epsilon):
    max = 0
    for i in range(n):
        for j in range(k):
            diff = np.abs(matrix[i, j] - old_matrix[i, j])
            if diff > max:
                max = diff
    return max < epsilon


def c_means():
    n = 100  # кол-во точек
    k = 4  # кол-во кластеров
    m = 1.5  # экс. вес
    epsilon = 0.1  # параметр останова

    x = np.random.randint(1, 100, n)
    y = np.random.randint(1, 100, n)

    matrix = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            matrix[i, j] = np.random.uniform(1, 4)

    while True:
        x_c, y_c = recntr(x, y, matrix, k, m)
        new_matrix = calculate_new_probability_clusters(n, k, x, y, x_c, y_c, m)
        if did_algo_converge(new_matrix, matrix, n, k, epsilon):
            clusters = cluster(new_matrix, n, k)
            draw(x, y, clusters, x_c, y_c)
            break
        matrix = new_matrix
    return x_c, y_c, new_matrix


if __name__ == '__main__':
    c_means()