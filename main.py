import cv2
from chessboard_segmentation import lines_detection, intersections_detection
from fields_crop import intersections_image_crop
import numpy as np
import glob2


def process_one_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = lines_detection.find_lines(gray)
    lines = lines_detection.filter_close_lines(lines)
    line_clusters = lines_detection.cluster_lines(lines)
    intersections = intersections_detection.find_intersections(line_clusters)
    centroids = intersections_detection.find_intersections_centroids(intersections, max(img.shape) * 0.033)
    intersections, centroids_sparse_matrix = intersections_detection.organize_intersection_centroids(centroids, max(
        img.shape) * 0.033)
    cropped_images = intersections_image_crop.crop_fields(img, centroids_sparse_matrix, (2, 1))
    return cropped_images, intersections, centroids_sparse_matrix


def put_intersections(img, intersections):
    ctn = 0
    for inter in intersections:
        color = (255, 0, 0)
        cv2.circle(img, tuple((np.round(inter).astype(int))), 8, tuple(color), 20)
        cv2.putText(img, str(ctn), tuple((np.round(inter).astype(int))), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
        ctn += 1


if __name__ == '__main__':
    for i in glob2.glob("dataset/**"):
        img = cv2.imread(i)
        cropped_image_parts, intersects, intersects_sparse_matrix = process_one_image(img)

        img_show = img.copy()
        put_intersections(img_show, intersects)
        cv2.imshow("chess board {}".format(i), cv2.resize(img_show, (1000, 1000)))
        for img2 in cropped_image_parts:
            cv2.imshow("cropped part {}".format(i), img2)
            cv2.waitKey(0)
            cv2.destroyWindow("cropped part {}".format(i))
        cv2.destroyAllWindows()
