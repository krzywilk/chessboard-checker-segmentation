import random

import cv2
from chessboard_segmentation import lines_detection, intersections_detection
from fields_crop import intersections_image_crop
import numpy as np
import glob2

from fields_crop.intersections_image_crop import show_crop_fields


def process_one_image(img, img2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines_ = lines_detection.find_lines(gray)
    lines = lines_detection.filter_close_lines(lines_)
    line_clusters = lines_detection.cluster_lines(lines)
    intersections = intersections_detection.find_intersections(line_clusters)
    centroids = intersections_detection.find_intersections_centroids(intersections, max(img.shape) * 0.033)
    intersections, centroids_sparse_matrix = intersections_detection.organize_intersection_centroids(centroids, max(
        img.shape) * 0.033)
    cropped_images = intersections_image_crop.crop_fields(img, centroids_sparse_matrix, (1, 1))
    return cropped_images, intersections, centroids_sparse_matrix, lines_


def put_intersections(img, intersections):
    ctn = 0
    for inter in intersections:
        color = (255, 0, 0)
        cv2.circle(img, tuple((np.round(inter).astype(int))), 8, tuple(color), 20)
        cv2.putText(img, str(ctn), tuple((np.round(inter).astype(int))), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
        ctn += 1


if __name__ == '__main__':
    ss = cv2.ximgproc.segmentation.createGraphSegmentation(sigma = 1,k = 300, min_size = 3)
    for i in glob2.glob("dataset/images/**"):
        if "\\4.jpg" not in i :#and "\\2.jpg" not in i and "\\13.jpg" not in i:
            continue
        img = cv2.imread(i)
        # img2 = cv2.imread(i.replace("images", "masks"))


        img_show = img.copy()
        # put_intersections(img_show, intersects)
        cv2.imshow("chess board {}".format(i), cv2.resize(img_show, (1000, 1000)))
        numShowRects = 100
        increment = 50


        cropped_image_parts, intersects, intersects_sparse_matrix, lines = process_one_image(img, img)



        show_crop_fields(img.copy(), img, intersects_sparse_matrix, (1, 1), lines)

        # cv2.imshow("edges", cv2.resize(edges, (1000, 1000)))
        # cv2.imshow("gray", cv2.resize(gray, (1000, 1000)))
        cv2.imshow("all", cv2.resize(img, (1000, 1000)))
        # cv2.waitKey(0)
        strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        # ss.addStrategy(strategy_color)
        for img2 in cropped_image_parts:


            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 11)
            x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5, scale=3)
            y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5, scale=3)
            absx = cv2.convertScaleAbs(x)
            absy = cv2.convertScaleAbs(y)
            edges = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
            (T, edges) = cv2.threshold(edges, 200, 255,
                                       cv2.THRESH_BINARY)
            # edges = cv2.Canny(final, 100, 200)
            # edges = cv2.erode(edges, np.ones((1,1)))
            # lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
            #                         minLineLength=50, maxLineGap=80)
            # if lines is not None:
            #     a, b, c = lines.shape
            #     for i in range(a):
            #         cv2.line(edges, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), 0, 3,
            #                  cv2.LINE_AA)

            edges = cv2.erode(edges, np.ones((5,5)), 1)
            edges = cv2.dilate(edges, np.ones((10,10)), 2)
            img2[edges ==0] = 255
            edges = cv2.Canny(img2, 100, 200)
            # img2_ = ss.processImage(img2)

            # img2_ = np.stack((img2_, img2_, img2_), 2)
            # for num in np.unique(img2_):
            #     img2_[:,:,0][img2_[:,:,0] == num] = len(img2_[:,:,0][img2_[:,:,0] == num]) * [random.randint(0, 256)]
            #     img2_[:,:,1][img2_[:,:,1] == num] = len(img2_[:,:,1][img2_[:,:,1] == num]) * [random.randint(0, 256)]
            #     img2_[:,:,2][img2_[:,:,2] == num] = len(img2_[:,:,2][img2_[:,:,2] == num]) * [random.randint(0, 256)]

            # cv2.imshow("cropped part {}".format(i), edges)
            # cv2.imshow("cropped wwpart {}".format(i), img2)
            # cv2.imshow("cropped wwspart {}".format(i), img2_.astype(np.uint8))
            #cv2.waitKey(0)
            # cv2.destroyWindow("cropped part {}".format(i))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
