import numpy as np
import cv2
from collections import defaultdict
from sklearn import preprocessing
from sklearn.cluster import DBSCAN


def find_lines(image, lower_canny=60, upper_canny=150, rho=1,
                 theta=np.pi / 180, threshold=200):
    """
    Finds lines on input image with usage of cv2.Canny and cv2.HoughLines
    :param image: grey input image
    :param lower_canny: cv2.Canny: threshold1 first threshold for the hysteresis procedure
    :param upper_canny:cv2.Canny: threshold2 second threshold for the hysteresis procedure
    :param rho: cv2.HoughLines: Distance resolution of the accumulator in pixels.
    :param theta: cv2.HoughLines: theta Angle resolution of the accumulator in radians.
    :param threshold: cv2.HoughLines: Accumulator threshold parameter. Only those lines are returned that get enough votes
    :return: Lines np.ndarray. Each cell has [rho, theta] values
    """
    edges = cv2.Canny(image, lower_canny, upper_canny)
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    return lines


def filter_close_lines(lines, atol_rho=10, atol_theta=np.pi / 36):
    """
    Filters lines close to each other.
    :param lines: Lines np.ndarray
    :param atol_rho: The absolute tolerance parameter used to filter close rho values of lines.
    :param atol_theta: The absolute tolerance parameter used to filter close theta values of lines.
    :return: Filtered lines np.ndarray
    """
    res_lines_alloc = np.zeros([len(lines), 1, 2])
    res_lines_alloc[0] = lines[0]
    end_alloc_pointer = 1
    for line in lines:
        rho, theta = line[0]
        if rho < 0:
            rho *= -1
            theta -= np.pi
        closeness_rho = np.isclose(rho, res_lines_alloc[0:end_alloc_pointer, 0, 0], atol=atol_rho)
        closeness_theta = np.isclose(theta, res_lines_alloc[0:end_alloc_pointer, 0, 1], atol=atol_theta)
        closeness = np.all([closeness_rho, closeness_theta], axis=0)
        if not any(closeness):
            res_lines_alloc[end_alloc_pointer] = line
            end_alloc_pointer += 1
    return res_lines_alloc[:end_alloc_pointer]


def cluster_lines(lines):
    """
    Clusters lines into vertical and horizontal groups with the DBSCAN algorithm. Lines recognized as an
    outliers (e.g. 45 degrees lines) are not returned
    :param lines: Lines np.ndarray
    :return: List of lines clusters
    """
    angles = np.array([line[0][1] for line in lines])
    pts = np.array([[np.cos(2 * theta), np.sin(2 * theta)]
                    for theta in angles], dtype=np.float32)
    pts = preprocessing.normalize(pts)
    db = DBSCAN(eps=0.2, min_samples=9).fit(pts)
    labels = db.labels_
    segmented = defaultdict(list)
    for line, i in zip(lines, range(len(lines))):
        if labels[i] > -1:
            segmented[labels[i]].append(line)
    return list(segmented.values())
