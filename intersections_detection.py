import numpy as np
import cv2
from collections import defaultdict

def kmeans_segment_vertical_horizontal_lines(lines : np.ndarray, k : int = 2, max_iter : int = 10, eps : int = 1, attempts : int = 10) -> list:
    any_condition_met = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    angles = np.array([line[0][1] for line in lines])
    pts = np.array([[np.cos(2 * theta), np.sin(2 * theta)]
                    for theta in angles], dtype=np.float32)
    labels, _ = cv2.kmeans(pts, k, None, (any_condition_met, max_iter, eps), attempts, cv2.KMEANS_RANDOM_CENTERS)[1:]
    labels = labels.reshape(-1)
    segmented = defaultdict(list)
    for line, i in zip(lines, range(len(lines))):
        segmented[labels[i]].append(line)
    return list(segmented.values())

def intersection(line1 : np.ndarray, line2 : np.ndarray) -> tuple:
    """src https://stackoverflow.com/a/383527/5087436"""
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

def find_intersections(segmented_lines : list) -> list:
    intersections = []
    first_group_lines = segmented_lines[0]
    second_group_lines = segmented_lines[1]
    for first_line in first_group_lines:
        for second_line in second_group_lines:
            intersections.append(intersection(first_line, second_line))
    return intersections

