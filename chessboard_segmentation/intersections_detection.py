import numpy as np
from sklearn.cluster import DBSCAN


def intersection(line1, line2):
    """
    Finds intersection point of two lines.

    src: https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    :param line1: First line array containing rho and theta
    :param line2: second line array containing rho and theta
    :return: x and y of intersection point
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return x0, y0


def find_intersections(lines_clusters):
    """
    Finds the intersections between groups of lines.

    src: https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    :param lines_clusters: List of clustered lines
    :return: np.ndarray of intersection points
    """
    intersections = []
    first_group_lines = lines_clusters[0]
    second_group_lines = lines_clusters[1]
    for first_line in first_group_lines:
        for second_line in second_group_lines:
            intersections.append(intersection(first_line, second_line))
    return np.array(intersections)


def find_intersections_centroids(intersections, eps, min_samples=1):
    """
    Clusters intersections with the DBSCAN algorithm, and compute mean point for each cluster.
    :param intersections: np.ndarray of intersection points
    :param eps: maximum distance between points
    :param min_samples: minimum samples in centroid cluster
    :return: np.ndarray of mean point for each centroid
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(intersections)
    labels = db.labels_
    centroids = np.array(
        [intersections[labels == i].mean(axis=0).astype(int) for i in range(np.min(labels), np.max(labels) + 1)])
    centroids = centroids[np.argsort(centroids[:, 1])]
    return centroids


def organize_intersection_centroids(intersections, eps):
    """
    Organizes intersections in rows and lines with correct order.
    :param intersections: np.ndarray of intersection points
    :param eps: max position difference between points in line
    :return:
        intersections: np.ndarray of intersection points organised from left to right and from top to the bottom
        centroids_sparse_matrix: sparse np.ndarray of intersection points organised in matrix
    """
    centroids_sparse_matrix = np.empty((len(intersections), len(intersections), 2))
    centroids_sparse_matrix[:] = np.nan
    pointer = 0
    row = 0
    max_col = 0
    for i in range(1, len(intersections) + 1):
        if i == len(intersections) or intersections[i][1] > intersections[i - 1][1] + eps:
            intersections[pointer: i] = intersections[pointer: i][np.argsort(intersections[pointer:i, 0])]
            centroids_sparse_matrix[row][0: i - pointer] = intersections[pointer: i]
            if max_col < i - pointer:
                max_col = i - pointer
            row += 1
            pointer = i
    return intersections, centroids_sparse_matrix
